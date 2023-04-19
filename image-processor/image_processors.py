import cv2
import inspect
import image_processors_helper as helper


def _convert_function_name_to_class_name(function_name):
    return "".join(word.capitalize() for word in function_name.split("_"))


def _get_registered_functions(module):
    functions = {}
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if hasattr(func, "image_procesor_info"):
            class_name = _convert_function_name_to_class_name(name)
            try:
                functions[class_name] = {
                    "func": func,
                    "input_amount": func.image_procesor_info[0],
                    "output_amount": func.image_procesor_info[1],
                    "default_values": func.default_variables if hasattr(func, "default_variables") else {},
                    "input_information": func.input_information ,
                    "output_information": func.output_information,
                    "user_input_information": func.user_input_information,
                }
            except Exception as e:
                print(e)
                print(class_name)
    return functions


_functions = _get_registered_functions(helper)

def generate_typed_helpers():
    rv = []
    for function_name, function_info in _functions.items():
        vars = function_info["default_values"]
        user_input = function_info["user_input_information"]
        txt1_1 = []
        txt1_2 = []
        txt2_1 = []
        txt2_2 = []
        for var in user_input.keys():
            if var in vars.keys():
                txt1_1.append(f'{var}: {user_input[var]} = {vars[var]}')
                txt2_1.append(f'{var} = {vars[var]} (type: {user_input[var]})\n')
            else:
                txt1_2.append(f'{var}: {user_input[var]}')
                txt2_2.append(f'{var} (type: {user_input[var]})\n')
        txt = ', '.join(txt1_2 + txt1_1)
        txt2 = '# '.join(txt2_1 + txt2_2)

        a = function_info['input_information'] 
        b = function_info['output_information'] 
        rv.append(f"""# This node takes object of type {a['image']} and returns object of type {b['image']}
# Total amount of data that comes in is {"unlimited" if function_info['input_amount'] == -1 else function_info['input_amount']}.
# Total amount of data that comes out is {"unlimited" if function_info['output_amount'] == -1 else function_info['output_amount']}.
# The default values are as follows:
# name (type: str)
# {txt2}
class {function_name}:
    def __init__(self, name,  {txt}):
        pass
""")
    return '\n'.join(rv)
def __getattr__(name):
    if name == "Get_Function_Keys":
        return _functions.keys()
    if name == "Get_Function_Values":
        return _functions.values()
    if name == "Get_Function_Items":
        return _functions.items()
    if name == "Get_Function_List":
        return _functions
    if name == "ImageProcessor":
        return helper.ImageProcessor
    if name == "generate_typed_helpers":
        return generate_typed_helpers

    if name in _functions:

        def wrapper(*args, **kwargs):
            kw = {**kwargs, **_functions[name]["default_values"]}
            for arg in zip(args, _functions[name]["user_input_information"]):
                if arg[0] is None:
                    kw[arg[1]] = _functions[name]["default_variables"][arg[1]]
                else:
                    kw[arg[1]] = arg[0]

            return helper.ImageProcessor(
                _functions[name]["func"],
                _functions[name]["input_amount"],
                _functions[name]["output_amount"],
                kw,
            )

        return wrapper
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")