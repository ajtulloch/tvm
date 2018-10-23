import resnet
import unet
import mxnet as mx

def get_mxnet_symbol(model, align):
    if model == "unet":
        sym = unet.unet(alignment=align)
        image_shape = (3, 192, 192)
        output_shape = (1, 192, 192)
    if model == "resnet50":
        sym = resnet.get_symbol(num_classes=1000, num_layers=50, image_shape="3, 224, 224")
        image_shape = (3, 224, 224)
        output_shape = (1000,)
    return sym, image_shape, output_shape

def get_nnvm_sym(sym, image_shape):
    import nnvm
    mod = mx.mod.Module(symbol=sym, context=mx.cpu())
    mod.bind(data_shapes=[('data', tuple([1] + list(image_shape)))])
    mod.init_params()
    sym, params = nnvm.frontend.from_mxnet(
        sym,
        arg_params=mod.get_params()[0],
        aux_params=mod.get_params()[1])
    return sym, params