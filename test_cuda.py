import torch
import PIL

print("\n")

print(torch.cuda.is_available())  # Should return True

print("\n")

print(torch.cuda.current_device())  # Should return the current device index

print("\n")

print(torch.cuda.get_device_name(0))  # Should return the name of your GPU

print("\n")

print(torch.cuda.device_count())

print("\n")

print(torch.cuda.get_arch_list())  # Should return the sm_x versions of your device architecture

print("\n")

print(torch.version.cuda)  # Should print '10.1'

print("\n")

print(torch.backends.cudnn.version())

print("\n")

print(torch.backends.cudnn.is_available())  # Should print True

print("\n")

print(PIL.__version__)