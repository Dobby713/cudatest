
def create_model(opt):
    model = None

    if opt.model == 'pix2pix_classifier':
        assert(opt.dataset_mode == 'unaligned')
        from .pix2pix_classifier_model import Pix2PixClassifierModel
        model = Pix2PixClassifierModel()

    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()

    else:
        raise ValueError("Model [%s] not recognized." % opt.model)

    model.initialize(opt)
    return model
