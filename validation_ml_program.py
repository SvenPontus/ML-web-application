

class Validation:
    """My Validation class"""
        
    @staticmethod
    def validate_yes_or_no(value:str):
        return value in ["yes", "y", "no", "n"]
        
    @staticmethod
    def validate_str(value:str):
        return isinstance(value, str)

    @staticmethod
    def validate_int(value:int):
        return isinstance(int(value), int)
    
    @staticmethod
    def validate_r_or_c(value:str):
        return value == "r" or value == "c"
    
    @staticmethod
    def controll_csv(value):   
        if value.endswith(".csv"):
            return value
              
#  -  -  -  -  -  -  -  User input validation  -  -  -  -  -  -  -  -  #
    
    @staticmethod
    def read_in_str_value(validation_function, message:str):
        while True:
            user_input = input(message)
            if validation_function(user_input):
                return user_input
    
    @staticmethod
    def read_in_int_value(validation_function, message:str):
        while True:
            user_input = (input(message))
            if user_input.isnumeric() and validation_function(int(user_input)):
                return int(user_input)

#  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  #




