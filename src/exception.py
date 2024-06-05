import sys

def exception_handler(exception, message:sys):
    _,_,exc_tb=message.exc_info()
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
     exc_tb.tb_frame.f_code.co_filename, exc_tb.tb_lineno, str(exception))
    
    return error_message

class Exception(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=exception_handler(error_message,message=error_detail)
    
    def __str__(self):
        return self.error_message