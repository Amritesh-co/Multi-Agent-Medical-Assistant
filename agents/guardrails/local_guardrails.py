from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

# LangChain Guardrails
class LocalGuardrails:
    """Guardrails implementation using purely local components with LangChain."""
    
    def __init__(self, llm):
        """Initialize guardrails with the provided LLM."""
        self.llm = llm
        
        # Input guardrails prompt
        self.input_check_prompt = PromptTemplate.from_template(
            """Evaluate if the user input is safe for a MEDICAL ASSISTANT.
            
            USER INPUT: {input}
            
            SAFE content includes:
            - Greetings and introductions
            - Medical questions about symptoms, diseases, and treatments
            - Requests for health information
            
            UNSAFE content includes:
            - Requests for illegal drugs or harmful instructions
            - Self-harm or suicide
            - Personal Private Information (PII)
            - Sexual harassment or explicit content
            
            Is this input SAFE? Respond ONLY with "SAFE" or "UNSAFE: [reason]".
            
            Decision:
            """
        )
        
        # Output guardrails prompt
        self.output_check_prompt = PromptTemplate.from_template(
            """TASK: Review this medical chatbot response.
            
            ORIGINAL QUERY: {user_input}
            CHATBOT RESPONSE: {output}
            
            INSTRUCTIONS:
            - If the response is safe, respond with ONLY the word "ORIGINAL".
            - If the response is unsafe or contains technical errors, provide a REVISED safe version.
            - DO NOT include any reasoning, analysis, or "Safety Review Result" in your response.
            
            Final Response (ORIGINAL or REVISED):
            """
        )
        
        # Create the input guardrails chain
        self.input_guardrail_chain = (
            self.input_check_prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        # Create the output guardrails chain
        self.output_guardrail_chain = (
            self.output_check_prompt 
            | self.llm 
            | StrOutputParser()
        )
    
    def check_input(self, user_input: str) -> tuple[bool, str]:
        """
        Check if user input passes safety filters.
        
        Args:
            user_input: The raw user input text
            
        Returns:
            Tuple of (is_allowed, message)
        """
        result = self.input_guardrail_chain.invoke({"input": user_input})
        
        if result.startswith("UNSAFE"):
            reason = result.split(":", 1)[1].strip() if ":" in result else "Content policy violation"
            return False, AIMessage(content = f"I cannot process this request. Reason: {reason}")
        
        return True, user_input
    
    def check_output(self, output: str, user_input: str = "") -> str:
        """
        Process the model's output through safety filters.
        
        Args:
            output: The raw output from the model
            user_input: The original user query (for context)
            
        Returns:
            Sanitized/modified output
        """
        if not output:
            return output
            
        # Convert AIMessage to string if necessary
        output_text = output if isinstance(output, str) else output.content
        
        result = self.output_guardrail_chain.invoke({
            "output": output_text,
            "user_input": user_input
        })
        
        # If the result is exactly "ORIGINAL" (case-insensitive), return the original text
        if result.strip().upper() == "ORIGINAL":
            return output_text
        
        return result