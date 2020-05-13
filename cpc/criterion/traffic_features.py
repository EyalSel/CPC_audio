import torch
import cpc.feature_loader as fl
from cpc.feature_loader import seqNormalization
from cpc.feature_loader import FeatureModule
import numpy as np
from tqdm import tqdm


# copied from cpc.feature_loader.buildFeature and modified for traffic
def buildFeature(featureMaker, seq, strict=False,
                 maxSizeSeq=64000, seqNorm=False, gpu_id=None):
    r"""
    Apply the featureMaker to the given file.
    Arguments:
        - featureMaker (FeatureModule): model to apply
        - seqPath (string): path of the sequence to load
        - strict (bool): if True, always work with chunks of the size
                         maxSizeSeq
        - maxSizeSeq (int): maximal size of a chunk
        - seqNorm (bool): if True, normalize the output along the time
                          dimension to get chunks of mean zero and var 1
    Return:
        a torch vector of size 1 x Seq_size x Feature_dim
    """
    print(seq)
    sizeSeq = seq.size(1)
    start = 0
    out = []
    pbar = tqdm(total=sizeSeq)
    while start < sizeSeq:
        if strict and start + maxSizeSeq > sizeSeq:
            break
        end = min(sizeSeq, start + maxSizeSeq)
        subseq = (seq[:, start:end]).view(1, 1, -1)
        if gpu_id is not None:
            subseq = subseq.cuda(device=gpu_id)
        with torch.no_grad():
            features = featureMaker((subseq, None))
            if seqNorm:
                features = seqNormalization(features)
        out.append(features.detach().cpu())
        start += maxSizeSeq
        pbar.update(maxSizeSeq)
    pbar.close()

    if strict and start < sizeSeq:
        subseq = (seq[:, -maxSizeSeq:]).view(1, 1, -1)
        if gpu_id is not None:
            subseq = subseq.cuda(device=gpu_id)
        with torch.no_grad():
            features = featureMaker((subseq, None))
            if seqNorm:
                features = seqNormalization(features)
        delta = (sizeSeq - start) // featureMaker.getDownsamplingFactor()
        out.append(features[:, -delta:].detach().cpu())

    out = torch.cat(out, dim=1)
    return out


def preprocess_traffic(checkpoint, input_trace, use_z, gpu_id=None):
    featureMaker = fl.loadModel([checkpoint], loadStateDict=True)[0]
    featureMaker = FeatureModule(featureMaker, use_z)
    featureMaker.collapse = False
    if gpu_id is not None:
        featureMaker = featureMaker.cuda(device=gpu_id)
    featureMaker.eval()

    seq = input_trace.astype(np.float32)
    seq = seq.reshape(1, -1, 1)
    seq = torch.tensor(seq)
    result = buildFeature(featureMaker, seq, strict=False, seqNorm=False, gpu_id=gpu_id)
    np.save("/shared_volume/output_embedding", result.data.numpy())
    print(result.shape)
