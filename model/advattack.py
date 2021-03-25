from art.attacks.evasion import ProjectedGradientDescent, FastGradientMethod
from art.estimators.object_detection import PyTorchFasterRCNN

def gen_adv_model(model):
    detector = PyTorchFasterRCNN(model=model, clip_values=(0, 1), preprocessing=None)
    return detector

def gen_adv_attack(model):
    attack = ProjectedGradientDescent(model, eps=0.01, eps_step=0.01, max_iter=2, verbose=True)
    return attack

