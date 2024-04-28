
import torch

def _create(name, pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    from pathlib import Path
    from models.yolo import Model, attempt_load
    from utils.general import check_requirements, set_logging
    from utils.google_utils import attempt_download
    from utils.torch_utils import select_device
    check_requirements(Path(__file__).parent / 'requirements.txt', exclude=('tensorboard', 'pycocotools', 'thop'))
    set_logging(verbose=verbose)

    fname = Path(name).with_suffix('.pt')  # checkpoint filename
    try:
        if pretrained and channels == 3 and classes == 80:
            model = attempt_load(fname, map_location=torch.device('cpu'))  # download/load FP32 model
        else:
            cfg = list((Path(__file__).parent / 'models').rglob(f'{name}.yaml'))[0]  # model.yaml path
            model = Model(cfg, channels, classes)  # create model
            if pretrained:
                ckpt = torch.load(attempt_download(fname), map_location=torch.device('cpu'))  # load
                msd = model.state_dict()  
                csd = ckpt['model'].float().state_dict() 
                csd = {k: v for k, v in csd.items() if msd[k].shape == v.shape}  # filter
                model.load_state_dict(csd, strict=False)  
                if len(ckpt['model'].names) == classes:
                    model.names = ckpt['model'].names  # set class names attribute
        if autoshape:
            model = model.autoshape() 
        device = select_device('0' if torch.cuda.is_available() else 'cpu') if device is None else torch.device(device)
        return model.to(device)

    except Exception as e:
        help_url = 'https://github.com/ultralytics/yolov5/issues/36'
        s = 'Cache may be out of date, try `force_reload=True`. See %s for help.' % help_url
        raise Exception(s) from e


def custom(path='path/to/model.pt', autoshape=True, verbose=True, device=None):
    return _create(path, autoshape=autoshape, verbose=verbose, device=device)

def yolov5s(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    return _create('yolov5s', pretrained, channels, classes, autoshape, verbose, device)


def yolov5m(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    return _create('yolov5m', pretrained, channels, classes, autoshape, verbose, device)


def yolov5l(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    return _create('yolov5l', pretrained, channels, classes, autoshape, verbose, device)


def yolov5x(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    return _create('yolov5x', pretrained, channels, classes, autoshape, verbose, device)


def yolov5s6(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    return _create('yolov5s6', pretrained, channels, classes, autoshape, verbose, device)


def yolov5m6(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    return _create('yolov5m6', pretrained, channels, classes, autoshape, verbose, device)


def yolov5l6(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    return _create('yolov5l6', pretrained, channels, classes, autoshape, verbose, device)


def yolov5x6(pretrained=True, channels=3, classes=80, autoshape=True, verbose=True, device=None):
    return _create('yolov5x6', pretrained, channels, classes, autoshape, verbose, device)


if __name__ == '__main__':
    model = _create(name='yolov5s', pretrained=True, channels=3, classes=80, autoshape=True, verbose=True)  # pretrained
    # model = custom(path='path/to/model.pt')  # custom

    # Verify inference
    import cv2
    import numpy as np
    from PIL import Image

    imgs = ['data/images/zidane.jpg',  # filename
            'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg',  # URI
            cv2.imread('data/images/bus.jpg')[:, :, ::-1],  # OpenCV
            Image.open('data/images/bus.jpg'),  # PIL
            np.zeros((320, 640, 3))]  # numpy

    results = model(imgs)  # batched inference
    results.print()
    results.save()
