"""The attributes node used for Relay operators"""

from ...attrs import Attrs
from ..base import register_relay_attr_node

@register_relay_attr_node
class Conv2DAttrs(Attrs):
    """Attribute of nn.conv2d"""


@register_relay_attr_node
class Conv2DWinogradAttrs(Attrs):
    """Attribute of nn.contrib_conv2d_winograd_without_weight_transform"""


@register_relay_attr_node
class Conv2DWinogradWeightTransformAttrs(Attrs):
    """Attribute of nn.contrib_conv2d_winograd_weight_transform"""


@register_relay_attr_node
class GlobalPool2DAttrs(Attrs):
    """Attribute of nn.global_pool"""

@register_relay_attr_node
class MaxPool2DAttrs(Attrs):
    """Attribute of nn.max_pool2d"""

@register_relay_attr_node
class UpSamplingAttrs(Attrs):
    """Attribute of nn.upsampling"""
