class SMPValidationError(ValueError):
    pass


class SMPConfigError(ValueError):
    # This is only raised when there is an smp bug, not a user error
    pass


class CommGroupConfigError(SMPValidationError):
    pass


class WorkerSizeError(SMPValidationError):
    pass


class LoggingConfigError(SMPValidationError):
    pass


class InvalidEnvironmentError(SMPValidationError):
    def __str__(self):
        return "SageMaker environment not found"


class NotInitializedError(SMPValidationError):
    def __str__(self):
        return "smp is not initialized!"


class InvalidTransactionIDError(SMPValidationError):
    def __init__(self, transaction_id):
        self.transaction_id = transaction_id

    def __str__(self):
        return f"Invalid transaction ID: {self.transaction_id}"


class InvalidLinkIDError(SMPValidationError):
    def __init__(self, link_id):
        self.link_id = link_id

    def __str__(self):
        return f"Link id {self.link_id} does not exist!"


class InvalidStepOutputError(SMPValidationError):
    def __init__(self, outputs_type):
        self.outputs_type = outputs_type

    def __str__(self):
        return f"StepOutput only accepts list or tuple, but get {self.outputs_type}"


class SMPInvalidArgumentError(SMPValidationError):
    pass


class SMPUnsupportedError(RuntimeError):
    pass


class SMPRuntimeError(RuntimeError):
    pass


class TensorSplitError(SMPRuntimeError):
    pass


class InitializationError(SMPRuntimeError):
    pass


class SMPSegFault(SMPRuntimeError):
    pass
