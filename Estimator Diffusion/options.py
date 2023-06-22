class TrainingOptions:
    """
    Configuration options for the training
    """

    def __init__(self,
                 batch_size: int,
                 number_of_epochs: int,
                 start_epoch: int):
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs
        self.start_epoch = start_epoch


class Configuration():
    """
    The HiDDeN network configuration.
    """

    def __init__(self, H: int, W: int, message_length: int,
                 encoder_blocks: int, encoder_channels: int,
                 decoder_blocks: int, decoder_channels: int,
                 use_discriminator: bool,
                 discriminator_blocks: int, discriminator_channels: int,
                 encoder_loss: float,
                 decoder_loss: float,
                 adversarial_loss: float,
                 enable_fp16: bool = False):
        self.H = H
        self.W = W
        self.message_length = message_length
        self.encoder_blocks = encoder_blocks
        self.encoder_channels = encoder_channels
        self.use_discriminator = use_discriminator
        self.decoder_blocks = decoder_blocks
        self.decoder_channels = decoder_channels
        self.discriminator_blocks = discriminator_blocks
        self.discriminator_channels = discriminator_channels
        self.encoder_loss = encoder_loss
        self.decoder_loss = decoder_loss
        self.adversarial_loss = adversarial_loss
        self.enable_fp16 = enable_fp16
