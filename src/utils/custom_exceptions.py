""" class to handle custom exceptions
"""

class CustomException(Exception):
    """Base class for other exceptions"""
    pass

class RetrieverSetupError(CustomException):
    """Raised when there is an error in setting up the retriever"""
    def __init__(self, message=None):
        self.message = "Failed to set up retriever" + message
        super().__init__(self.message)

class LLMSetupError(CustomException):
    """Raised when there is an error in setting up the LLM"""
    def __init__(self, message=None):
        self.message = "Failed to set up LLM" + message
        super().__init__(self.message)

class RAGChainSetupError(CustomException):
    """Raised when there is an error in setting up the RAG chain"""
    def __init__(self, message=None):
        self.message = message + "Failed to set up RAG chain"
        super().__init__(self.message)

class ConversationChainSetupError(CustomException):
    """Raised when there is an error in setting up the conversation chain"""
    def __init__(self, message=None):
        self.message = "Failed to set up conversation chain" + message
        super().__init__(self.message)
        

# Retriever Custom Exceptions
class DocuemntLoaderException(RetrieverSetupError):
    """Raised when there is an error in loading the document"""
    def __init__(self, message="Failed to load document"):
        self.message = message
        super().__init__(self.message)
class DocumentSplitterException(RetrieverSetupError):
    """Raised when there is an error in splitting the document"""
    def __init__(self, message="Failed to split document"):
        self.message = message
        super().__init__(self.message)

class DocumentEmbeddingException(RetrieverSetupError):
    """Raised when there is an error in embedding the document"""
    def __init__(self, message="Failed to embed/store document"):
        self.message = message
        super().__init__(self.message)
class DocumentRetrieverException(RetrieverSetupError):
    """Raised when there is an error in retrieving the document"""
    def __init__(self, message="Failed to retrieve document"):
        self.message = message
        super().__init__(self.message)
        
class GetDocumentRetrieverException(RetrieverSetupError):
    """Raised when there is an error in getting the document retriever"""
    def __init__(self, message="Failed to get document retriever"):
        self.message = message
        super().__init__(self.message)
        
# ConversationChainSetup Custom Exceptions
class TokenGenerationException(ConversationChainSetupError):
    """Raised when there is an error in generating token"""
    def __init__(self, message="Failed to generate token"):
        self.message = message
        super().__init__(self.message)