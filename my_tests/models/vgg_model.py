from vgg import cifar10vgg


def get_vgg_model(logits=False, input_ph=None):
    """
    Defines a CNN model using Keras sequential model
    :param logits: If set to False, returns a Keras model, otherwise will also
                    return logits tensor
    :param input_ph: The TensorFlow tensor for the input
                    (needed if returning logits)
                    ("ph" stands for placeholder but it need not actually be a placeholder)
    :return:
    """
    vgg = cifar10vgg(train=False)
    model = vgg.model

    if logits:
        logits_tensor = model(input_ph)
        return model, logits_tensor

    return model
