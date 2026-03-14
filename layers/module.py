from tensors import Tensor


class BaseModule:

    def parameters(self):
        params = []

        for value in self.__dict__.values():

            if isinstance(value, Tensor):
                params.append(value)

            elif isinstance(value, BaseModule):
                params.extend(value.parameters())

            elif isinstance(value, (list, tuple)):
                for v in value:

                    if isinstance(v, Tensor):
                        params.append(v)

                    elif isinstance(v, BaseModule):
                        params.extend(v.parameters())

        return params
    
    def zero_grad(self):
        for p in self.parameters():
            p.reset_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError
    

class Sequential(BaseModule):

    def __init__(self, *modules) -> None:
        super().__init__()
        self.modules = modules
    
    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        
        return x