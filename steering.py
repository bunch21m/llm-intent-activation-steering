import logging
import os
from typing import Dict, TypedDict, Annotated, Sequence, List, Tuple

import numpy as np
import torch
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_mistralai import ChatMistralAI
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure module logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ---- Local Model Wrapper ----

class SteeredModelWrapper:
    """
    A wrapper around a HuggingFace causal language model that supports dynamic
    activation steering via PyTorch forward hooks.
    """
    
    def __init__(self, model_name: str = "google/gemma-2-2b-it"):
        """
        Initializes the model wrapper, loads the tokenizer, determines the optimal
        device (CUDA, MPS, or CPU), and loads the model into memory.
        
        Args:
            model_name (str): The HuggingFace hub ID of the model to load.
        """
        logger.info(f"Loading local model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Check for optimal device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        logger.info(f"Using device: {self.device}")
        
        # Load model with appropriate precision
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if self.device != "cpu" else None,
            low_cpu_mem_usage=True
        )
        
        if self.device == "cpu":
            self.model.to("cpu")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Ensure a chat template exists (fallback for older models)
        if not self.tokenizer.chat_template:
            self.tokenizer.chat_template = (
                "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\\n' }}"
                "{% endfor %}{{ 'assistant: ' }}"
            )
            
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.active_steering: Dict[str, str] = {}
        self.last_active_steering: Dict[str, str] = {}

        # Pre-load steering vectors to avoid I/O bottlenecks during generation
        self.vectors = {}
        vector_paths = {
            "uncertainty": "./vectors/uncertainty_steering_vector.npy",
            "joyful": "./vectors/joyful_steering_vector.npy",
            "cat": "./vectors/cat_steering_vector.npy"
        }
        for name, path in vector_paths.items():
            if os.path.exists(path):
                self.vectors[name] = torch.tensor(np.load(path))
            else:
                logger.warning(f"Vector file not found at {path}")

    def register_hook(self, layer_idx: int, vector: torch.Tensor, coeff: float = 1.0) -> None:
        """
        Registers a PyTorch forward hook to inject a steering vector into a specific layer.
        
        Args:
            layer_idx (int): The index of the transformer layer to hook into.
            vector (torch.Tensor): The concept vector to inject.
            coeff (float): The scaling coefficient applied to the vector.
        """
        def hook_fn(module: torch.nn.Module, input_tensors: Tuple[torch.Tensor, ...], output: Tuple[torch.Tensor, ...] | torch.Tensor) -> Tuple[torch.Tensor, ...] | torch.Tensor:
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Ensure the vector is on the correct device and dtype
            steering_vec = vector.to(hidden_states.device).to(hidden_states.dtype)
            
            # Inject the latent intent: activation steering by simple addition
            hidden_states = hidden_states + (steering_vec * coeff)
            
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            return hidden_states

        # Attempt to find the specific transformer block depending on model architecture
        try:
            if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
                 layer = self.model.transformer.h[layer_idx]
            elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                 layer = self.model.model.layers[layer_idx]
            else:
                 # Fallback attempt
                 layer = getattr(self.model, "model").layers[layer_idx]
            
            handle = layer.register_forward_hook(hook_fn)
            self.hooks.append(handle)
            logger.info(f"Successfully registered forward hook at layer {layer_idx}")
        except (AttributeError, IndexError) as e:
            logger.error(f"Failed to find layer {layer_idx} in model architecture: {e}")

    def clear_hooks(self) -> None:
        """Removes all currently active forward hooks from the model."""
        logger.info("Clearing all active steering hooks.")
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.active_steering.clear()

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        """
        Mimics the LangChain invoke interface, generating a response based on chat history.
        
        Args:
            messages (List[BaseMessage]): The conversation history.
            
        Returns:
            AIMessage: The generated response from the language model.
        """
        chat_history = []
        system_prompt = ""
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_prompt += str(msg.content) + "\n\n"
                continue
            
            role = "user"
            if isinstance(msg, AIMessage):
                role = "assistant"
            
            content = str(msg.content)
            if role == "user" and system_prompt:
                content = system_prompt + content
                system_prompt = ""
                
            if chat_history and chat_history[-1]["role"] == role:
                chat_history[-1]["content"] += "\n\n" + content
            else:
                chat_history.append({"role": role, "content": content})
                
        if system_prompt:
            if chat_history and chat_history[-1]["role"] == "user":
                chat_history[-1]["content"] += "\n\n" + system_prompt.strip()
            else:
                chat_history.append({"role": "user", "content": system_prompt.strip()})

        prompt_text = self.tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        try:
            self.last_active_steering = dict(self.active_steering)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
        finally:
            self.clear_hooks()
        
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return AIMessage(content=response_text)

# Initialize global model instance (Singleton pattern for orchestrator efficiency)
try:
    response_model = SteeredModelWrapper()
except Exception as e:
    logger.error(f"Failed to initialize SteeredModelWrapper: {e}")
    raise

# ---- Tools ----

@tool
def lower_confidence_tool() -> str:
    """Invoked to steer the model away from overconfidence."""
    if "uncertainty" in response_model.vectors:
        vec = response_model.vectors["uncertainty"]
        response_model.register_hook(layer_idx=20, vector=vec, coeff=95.0)
        logger.info("Applied Lower Confidence steering vector.")
        response_model.active_steering["uncertainty"] = "applied"
    else:
        logger.warning("Lower Confidence vector not loaded.")
    return "Steering applied: Lower Confidence (Layer 20)"

@tool
def joyful_tool() -> str:
    """Invoked to steer the model towards a joyful tone."""
    if "joyful" in response_model.vectors:
        vec = response_model.vectors["joyful"]
        response_model.register_hook(layer_idx=20, vector=vec, coeff=95.0)
        logger.info("Applied Joyful steering vector.")
        response_model.active_steering["joyful"] = "applied"
    else:
        logger.warning("Joyful vector not loaded.")
    return "Steering applied: Joyful (Layer 20)"

@tool
def mention_cat_tool() -> str:
    """Invoked to steer the model towards talking about cats."""
    if "cat" in response_model.vectors:
        vec = response_model.vectors["cat"]
        response_model.register_hook(layer_idx=20, vector=vec, coeff=130.0)
        logger.info("Applied Mention Cat steering vector.")
        response_model.active_steering["cat"] = "applied"
    else:
        logger.warning("Mention Cat vector not loaded.")
    return "Steering applied: Mention Cat (Layer 20)"

@tool
def surprise_me_tool() -> str:
    """Invoked to steer the model with a random noise vector for a surprising effect."""
    # Generating a random noise vector for demonstration
    vec = torch.randn(response_model.model.config.hidden_size) * 0.5
    response_model.register_hook(layer_idx=20, vector=vec, coeff=80.0)
    logger.info("Applied Surprise Me (synthetic noise) steering vector.")
    response_model.active_steering["surprise"] = "applied"
    return "Steering applied: Surprise Me (Layer 20)"

@tool
def reset_steering_tool() -> str:
    """Resets the model to its default state by removing all hooks."""
    response_model.clear_hooks()
    return "Steering reset: Model hooks cleared."

# ---- Graph State & Router ----

class AgentState(TypedDict):
    """The state dictionary passed through the LangGraph execution graph."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState) -> str:
    """Determines whether the graph should invoke tools or proceed to generation."""
    last_message = state["messages"][-1]
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return "end"
    else:
        return "continue"

tools = [lower_confidence_tool, joyful_tool, mention_cat_tool, surprise_me_tool, reset_steering_tool]

# Router model (external API model, not steered)
# Uses Mistral to intelligently decide if intent-based steering is required.
try:
    router_model = ChatMistralAI(
        model="mistral-small-latest",
        timeout=500.0,
        max_retries=3
    ).bind_tools(tools)
except Exception as e:
    logger.warning(f"Could not initialize ChatMistralAI router (Check API keys): {e}")
    router_model = None

def router_call(state: AgentState) -> Dict[str, BaseMessage]:
    """Graph node: Uses ChatMistralAI to decide which steering tools to invoke."""
    if not router_model:
        logger.error("Router model is not initialized. Cannot perform intelligent routing.")
        return {"messages": AIMessage(content="Error: Router model not configured.")}
        
    system_prompt = SystemMessage(
        content="You are a highly intelligent intent-routing assistant. Your ONLY goal is to analyze the user's latest message and decide if and how to steer the underlying response model using the provided tools.\n"
                "Act on BOTH explicit and implicit intent:\n"
                "- If the user expresses sadness or frustration, use 'joyful_tool' to cheer them up.\n"
                "- If the user asks a highly ambiguous, unanswerable, or philosophical question, use 'lower_confidence_tool' to ensure a nuanced response.\n"
                "- If the user's tone is extremely informal (e.g., 'hey bro', 'sup'), use 'casual_tone_tool'.\n"
                "- If the user explicitly asks to talk about cats or says 'this is a test', use 'mention_cat_tool'.\n"
                "- If the user asks for a normal response or to reset, use 'reset_steering_tool'.\n"
                "If no steering is required, DO NOT call any tools, so the response model can answer normally.\n"
                f"Current steering status: {response_model.active_steering}"
    )
    
    response = router_model.invoke([system_prompt] + list(state["messages"]))
    return {"messages": response}

def response_call(state: AgentState) -> Dict[str, BaseMessage]:
    """Graph node: Generates the final response using the locally steered model."""
    system_prompt = SystemMessage(content="You are a helpful assistant. Respond to the user's query.")
    
    filtered_messages = [system_prompt]
    for m in state["messages"]:
        if isinstance(m, (HumanMessage, AIMessage, SystemMessage)) and not isinstance(m, ToolMessage):
             # Exclude the router's tool-calling AIMessage to avoid confusing the local model
             if isinstance(m, AIMessage) and hasattr(m, 'tool_calls') and m.tool_calls:
                 continue
             filtered_messages.append(m)
             
    response = response_model.invoke(messages=filtered_messages)
    return {"messages": response}

# Construct the LangGraph workflow
graph = StateGraph(AgentState)
graph.add_node("router_call", router_call)
graph.add_node("response_call", response_call)
graph.add_node("tool_node", ToolNode(tools=tools))

graph.add_edge(START, "router_call")

graph.add_conditional_edges("router_call", should_continue,
    {
        "continue": "tool_node",    
        "end": "response_call"
    })

graph.add_edge("tool_node", "router_call")
graph.add_edge("response_call", END)

app = graph.compile()