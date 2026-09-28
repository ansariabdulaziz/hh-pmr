import torch
from torch.utils.data import Dataset


class PatientDataset(Dataset):
    def __init__(self, patients):
        self.patients = [p for p in patients if len(p) > 0]

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        return idx, self.patients[idx]


def make_collate(num_meds, diag_pad, proc_pad):
    def collate(batch):
        patient_ids = [item[0] for item in batch]
        patients = [item[1] for item in batch]
        batch_size = len(patients)
        max_visits = max(len(p) for p in patients)
        max_diag = max(1, max(len(v[0]) for p in patients for v in p))
        max_proc = max(1, max(len(v[1]) for p in patients for v in p))

        diag_ids = torch.full((batch_size, max_visits, max_diag), diag_pad, dtype=torch.long)
        diag_mask = torch.zeros((batch_size, max_visits, max_diag), dtype=torch.bool)
        proc_ids = torch.full((batch_size, max_visits, max_proc), proc_pad, dtype=torch.long)
        proc_mask = torch.zeros((batch_size, max_visits, max_proc), dtype=torch.bool)
        med_multihot = torch.zeros((batch_size, max_visits, num_meds), dtype=torch.float)
        visit_mask = torch.zeros((batch_size, max_visits), dtype=torch.bool)
        visit_index = torch.zeros((batch_size, max_visits), dtype=torch.long)
        patient_id = torch.zeros((batch_size, max_visits), dtype=torch.long)

        for b, patient in enumerate(patients):
            for t, visit in enumerate(patient):
                diagnoses, procedures, medications = visit[0], visit[1], visit[2]
                if len(diagnoses) > 0:
                    diag_ids[b, t, :len(diagnoses)] = torch.tensor(diagnoses, dtype=torch.long)
                    diag_mask[b, t, :len(diagnoses)] = True
                if len(procedures) > 0:
                    proc_ids[b, t, :len(procedures)] = torch.tensor(procedures, dtype=torch.long)
                    proc_mask[b, t, :len(procedures)] = True
                valid = [m for m in medications if 0 <= m < num_meds]
                if len(valid) > 0:
                    med_multihot[b, t, torch.tensor(valid, dtype=torch.long)] = 1.0
                visit_mask[b, t] = True
                visit_index[b, t] = t
                patient_id[b, t] = patient_ids[b]

        return {
            "diag_ids": diag_ids,
            "diag_mask": diag_mask,
            "proc_ids": proc_ids,
            "proc_mask": proc_mask,
            "med_multihot": med_multihot,
            "visit_mask": visit_mask,
            "visit_index": visit_index,
            "patient_id": patient_id,
        }

    return collate
