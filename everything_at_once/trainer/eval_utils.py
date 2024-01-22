import torch
import torch.nn as nn
import numpy as np
from everything_at_once.model.utils.utils import sim_matrix, normalize_embeddings


def compute_sim_matrix(model, data, batch_size=16, sim='t_va'):
    assert sim == 't_va'
    sims = []

    for i in range(int(np.ceil(len(data['video']) / batch_size))):
        new_data = {}
        for field in ['video', 'video_mask', 'audio', 'audio_mask', 'nframes', 'audio_STFT_nframes']:
            if field in data:
                new_data[field] = data[field][i * batch_size: (i + 1) * batch_size]
        for field in ['text', 'text_mask']:
            new_data[field] = data[field]

        embeds = model(new_data, force_cross_modal=True)
        cur_sims = sim_matrix(normalize_embeddings(embeds['text_embed']), normalize_embeddings(embeds['v+a_embed'])).detach().cpu()
        sims.append(cur_sims)
    sims = torch.cat(sims, dim=1)
    return sims.permute(1, 0)


def print_result(values, logger=None, name='Recall'):
    for task, rec in values.items():
        if logger:
            logger.info(f'Task {task}. Recall = {rec * 100:0.1f}')
        print(f'Task {task}. Recall = {rec * 100:0.1f}')
    avg_recall = np.mean(list(values.values()))
    print('Recall: {0:0.1f}'.format(avg_recall * 100))


