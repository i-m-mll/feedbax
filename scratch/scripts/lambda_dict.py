# ---
# jupyter:
#   jupytext:
#     formats: ipynb,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: fx
#     language: python
#     name: python3
# ---

# %%
from abc import abstractmethod
import ast
from collections import OrderedDict
from collections.abc import MutableMapping
import dis
import inspect
from typing import Callable, TypeVar

from feedbax.utils import unzip2

# %% [markdown]
# Lambdas as hashed by their memory address, not their code.
#
# This means that we can't use them as keys, naively.

# %%
ldict = {
    lambda x: x: None,
    lambda x: x ** 2: False,
    lambda y: y: True,
}

ldict[lambda x: x]

# %% [markdown]
# However, for certain use cases this may be desirable behaviour.
#
# For example, when a task object specifies values for certain state variables during model initialization, it is useful to specify them by referring to subsets of an `AbstractState` object. And it is convenient to do this with lambdas, like `lambda state: state.mechanics.effector` rather than defining some `effector_substate` function somewhere, once-and-for-all. 
#
# These kinds of lambdas are very simple and comparing their code is a reliable way of seeing whether they are identical. Though, I think comparing bytecode might be better since we wouldn't want to force the user to use a particular parameter name (e.g. `state`). 
#
# Generally, we'll only be assigning to each substate once at most, so if we could use lambdas as dict keys, their uniqueness is OK.
#
# So, can we define a `LambdaDict` that simply alters the hashing behaviour for lambdas and allows them to be used as keys?

# %% [markdown]
# First, how can we compare the lambda code?

# %%
a = lambda x: x.a.b
b = lambda x: x.a.b.c
c = lambda y: y.a.b
d = lambda y: y 

# %% [markdown]
# Inspect will only work if we assume the bound variables have fixed names, or can eliminate them somehow.

# %%

print(inspect.getsource(a))
print(inspect.getsource(c))

# %% [markdown]
# But apparently the bound variable names also appear in the bytecode, so this isn't any better.

# %%
print(dis.dis(a))
print(dis.dis(c))


# %% [markdown]
# It looks like we'll need to deal with the bound variable problem in any case. So perhaps `InitSpecDict` should assume that all lambdas will have the form `lambda <x>: <x>.substate1.substate2.<etc>`, and simply hash everything after the second `<x>`.
#
# However, this is not straightforward with `inspect.getsource` as it doesn't just give the body of the lambda:

# %%
class TestInspect(OrderedDict):
    def __getitem__(self, key):
        print("get key source: ", inspect.getsource(key))
        return OrderedDict.__getitem__(self, key)
    
    def __setitem__(self, key, value):
        print("set key source: ", inspect.getsource(key))
        return OrderedDict.__setitem__(self, key, value)


# %%
a = TestInspect({lambda x: x: 3})

a[lambda x: x]

# %% [markdown]
# There is one more possibility: using `ast` to parse the inspected source, retrieving the attribute objects, and using them to construct a string:

# %%
where = lambda state: state.mechanics.plant.skeleton
where = lambda state: state

tree = ast.parse(inspect.getsource(where))
print(ast.dump(tree, indent=2))

# %%
lambda_ast = next((node for node in ast.walk(tree)
                   if isinstance(node, ast.Lambda)), None)

print(ast.dump(lambda_ast, indent=2))

# %%
attr_ast = next((node for node in ast.walk(lambda_ast) if isinstance(node, ast.Attribute)), None)
print(ast.dump(attr_ast, indent=2))

# %%
attr_obj = attr_ast
key = attr_ast.attr

while isinstance((attr_value := getattr(attr_obj, "value", None)), ast.Attribute):
    attr_obj = attr_value
    key = f"{attr_value.attr}.{key}"
    
key


# %% [markdown]
# Now, put this all into a function that takes a lambda and returns a unique string identifier:

# %%
def get_where_str(where_func):
    func_source = inspect.getsource(where_func).strip()
    func_ast = next((node for node in ast.walk(ast.parse(func_source))
                     if isinstance(node, ast.Lambda)), None)
    attr_ast = next((node for node in ast.walk(func_ast) 
                     if isinstance(node, ast.Attribute)), None)
    if attr_ast is None:
        return ""

    where_str = attr_ast.attr
    attr_obj = attr_ast
    
    while isinstance((attr_value := getattr(attr_obj, "value", None)), ast.Attribute):
        attr_obj = attr_value
        where_str = f"{attr_value.attr}.{where_str}"
    
    return where_str


# %%
where = lambda state: state.mechanics.plant.skeleton

get_where_str(where)


# %% [markdown]
# Now we can try to define our dictionary subclass

# %%
class InitSpecDict(OrderedDict):
    """
    """
    
    def __init__(self, *args, **kwargs):
        self._dict = OrderedDict()
        self.update(*args, **kwargs)
    
    def update(self, *args, **kwargs):
        for k, v in OrderedDict(*args, **kwargs).items():
            self[k] = v
            
    def __getitem__(self, key):
        where_str = get_where_str(key)
        return OrderedDict.__getitem__(self._dict, where_str)

    def __setitem__(self, key, value):
        where_str = get_where_str(key)
        self._dict[where_str] = value
    
    def __repr__(self):
        return "smee"


# %%
init_spec = InitSpecDict({
    lambda x: x.mechanics.effector: 3,
    lambda state: state.mechanics.plant.skeleton: None,
})

# %% [markdown]
# So the issue here is that `ast` expects valid python code strings, but `inspect.getsource` is chopping out a line from inside a dictionary definition.

# %%
ast.parse("    x = 1")

# %%
ast.parse("lambda x: x: 3,")

# %% [markdown]
# So finally, it may be necessary to generate the where string from bytecode.

# %%
bytecode = dis.Bytecode(where)
for instr in bytecode:
    print(instr.argrepr)


# %%
def get_where_str(where_func):
    bytecode = dis.Bytecode(where_func)
    return '.'.join(instr.argrepr for instr in bytecode
                    if instr.opname == "LOAD_ATTR")

get_where_str(where)

# %%
init_spec = InitSpecDict({
    lambda y: y: object,
    lambda x: x.mechanics.effector: 3,
    lambda state: state.mechanics.plant.skeleton: None,
})

# %%
init_spec[lambda state: state.mechanics.effector]

# %%
init_spec[lambda x: x]

# %% [markdown]
# Nice! This now works, even if it is questionable whether using `dis` is good practice...

# %% [markdown]
# However, some of the features of dicts do not work, because we're not subclassing `OrderedDict` correctly:

# %%
init_spec.items()


# %% [markdown]
# We can instead implement the interface to a dict by using the `MutableMapping` ABC.
#
# I'd prefer to use the `Mapping` ABC but I'll have to restructure `__init__` to avoid using `update`

# %%
def get_where_str(where_func):
    bytecode = dis.Bytecode(where_func)
    return '.'.join(instr.argrepr for instr in bytecode
                    if instr.opname == "LOAD_ATTR")


# %%
T = TypeVar('T')

class TransformedDict(MutableMapping[str, T]):
    """An `OrderedDict1 which transforms the keys, but also stores the original keys.
       
    Based on https://stackoverflow.com/a/3387975
    """
    
    def __init__(self, *args, **kwargs):
        self.store = OrderedDict()
        self.update(OrderedDict(*args, **kwargs))
            
    def __getitem__(self, key):
        # print('get key: ', key)
        # print('get: ', self._key_transform(key))
        return self.store[self._key_transform(key)][1]

    def __setitem__(self, key, value):
        # print('set: ', self._key_transform(key))
        self.store[self._key_transform(key)] = (key, value)
    
    def __delitem__(self, key):
        del self.store[self._key_transform(key)]
        
    def __iter__(self):
        # Apparently you're supposed to only yield the key
        for key in self.store:
            yield self.store[key][0]

    def __len__(self):
        return len(self.store)
    
    def __repr__(self):
        items_str = ', '.join(
            f"(lambda state: state{'.' if k else ''}{k}, {v})" 
            for k, (_, v) in self.store.items()
        )
        return f"{type(self).__name__}([{items_str}])"
    
    def to_str_dict(self):
        return {k: v for k, (_, v) in self.store.items()}
    
    @abstractmethod
    def _key_transform(self, key):
        ...
        
    def tree_flatten(self):
        """The same flatten function used by JAX for `dict`"""
        return unzip2(sorted(self.items()))[::-1]

    @classmethod
    def tree_unflatten(cls, keys, values):
        return cls(zip(keys, values))

class InitSpecDict(TransformedDict):
    def _key_transform(self, key):
        if isinstance(key, Callable):
            return get_where_str(key)
        return key


# %%
init_spec = InitSpecDict({
    lambda y: y: object,
    lambda x: x.mechanics.effector: 3,
    lambda state: state.mechanics.plant.skeleton: None,
})

# %%
init_spec[lambda x: x.mechanics.effector]

# %%
list(init_spec.items())

# %%
list(init_spec.keys())

# %%
list(init_spec.values())

# %%
print(init_spec)

# %%

# %% [markdown]
# How's the performance?

# %% [markdown]
# Construction:

# %%
# %%timeit 

InitSpecDict({
    lambda y: y: object,
    lambda x: x.mechanics.effector: 3,
    lambda state: state.mechanics.plant.skeleton: None,
    lambda state: state.fee: None,
    lambda state: state.fee1: None,
    lambda state: state.fee2: None,
    lambda state: state.fee3: None,
})

# %%
# %%timeit 

OrderedDict({
    '': object,
    'mechanics.effector': 3,
    'mechanics.plant.skeleton': None,
    'fee': None,
    'fee1': None,
    'fee2': None,
    'fee3': None,
})

# %% [markdown]
# Access:

# %%
init_spec = InitSpecDict({
    # lambda y: y: object,
    lambda x: x.mechanics.effector: 3,
    lambda state: state.mechanics.plant.skeleton: None,
    lambda state: state.fee: None,
    lambda state: state.fee1: None,
    lambda state: state.fee2: None,
    lambda state: state.fee3: None,
})

init_spec_simple = OrderedDict(init_spec.to_str_dict())

# %%
# %%timeit 

init_spec[lambda x: x.mechanics.effector]

# %%
# %%timeit

init_spec_simple['mechanics.effector']

# %% [markdown]
# This is almost entirely due to the cost of string transformation, as expected:

# %%
# %%timeit 

get_where_str(lambda x: x.mechanics.effector)

# %% [markdown]
# Get all items:

# %%
# %%timeit 

list(init_spec.items())

# %%
# %%timeit 

list(init_spec_simple.items())

# %%
