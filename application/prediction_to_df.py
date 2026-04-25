import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm 
import argparse

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--export", default="both",
                        type=str,
                        choices=["both", "aq", "event"])
    args = parser.parse_args()

    if args.export in ["both", "aq"]:
        PAQ_dir = os.path.join(os.getcwd(), 'SoundAQnet_scene_ISOPl_ISOEv_PAQ8DAQs')
        print(f"load from {PAQ_dir}")
        files = [p for p in Path(PAQ_dir).iterdir() if p.is_file()]

        dic = {
            'id': [],
            'ISOEvs': [],
            'ISOPls': [],
            'pleasant': [], 
            'eventful': [], 
            'chaotic': [], 
            'vibrant': [], 
            'uneventful': [], 
            'calm': [], 
            'annoying': [], 
            'monotonous': []
        }

        for i in tqdm(range(len(files)), total=len(files)):
            id = Path(files[i]).name.split('_scene_PAQ.txt')[0]
            df = pd.read_csv(files[i], sep=' ')
            ISOEvs, ISOPls = df.iloc[0,0].split('	')
            pleasant, eventful, chaotic, vibrant, uneventful, calm, annoying, monotonous = df.iloc[1,0].split('	')

            dic['id'] += [id]
            dic['ISOPls'] += [float(ISOPls)]
            dic['ISOEvs'] += [float(ISOEvs)]
            dic['pleasant'] += [float(pleasant)]
            dic['eventful'] += [float(eventful)]
            dic['chaotic'] += [float(chaotic)]
            dic['vibrant'] += [float(vibrant)]
            dic['uneventful'] += [float(uneventful)]
            dic['calm'] += [float(calm)]
            dic['annoying'] += [float(annoying)]
            dic['monotonous'] += [float(monotonous)]

        out = pd.DataFrame(dic)
        out_stats = out.groupby('id', as_index=True, group_keys=True).agg(
            min_ISOPls=('ISOPls', 'min'),
            max_ISOPls=('ISOPls', 'max'),
            std_ISOPls=('ISOPls', 'std'),
            avg_ISOPls=('ISOPls', 'mean'),

            min_ISOEvs=('ISOEvs', 'min'),
            max_ISOEvs=('ISOEvs', 'max'),
            std_ISOEvs=('ISOEvs', 'std'),
            avg_ISOEvs=('ISOEvs', 'mean'),

            min_pleasant=('pleasant', 'min'),
            max_pleasant=('pleasant', 'max'),
            std_pleasant=('pleasant', 'std'),
            avg_pleasant=('pleasant', 'mean'),

            min_eventful=('eventful', 'min'),
            max_eventful=('eventful', 'max'),
            std_eventful=('eventful', 'std'),
            avg_eventful=('eventful', 'mean'),

            min_chaotic=('chaotic', 'min'),
            max_chaotic=('chaotic', 'max'),
            std_chaotic=('chaotic', 'std'),
            avg_chaotic=('chaotic', 'mean'),

            min_vibrant=('vibrant', 'min'),
            max_vibrant=('vibrant', 'max'),
            std_vibrant=('vibrant', 'std'),
            avg_vibrant=('vibrant', 'mean'),

            min_uneventful=('uneventful', 'min'),
            max_uneventful=('uneventful', 'max'),
            std_uneventful=('uneventful', 'std'),
            avg_uneventful=('uneventful', 'mean'),

            min_calm=('calm', 'min'),
            max_calm=('calm', 'max'),
            std_calm=('calm', 'std'),
            avg_calm=('calm', 'mean'),

            min_annoying=('annoying', 'min'),
            max_annoying=('annoying', 'max'),
            std_annoying=('annoying', 'std'),
            avg_annoying=('annoying', 'mean'),

            min_monotonous=('monotonous', 'min'),
            max_monotonous=('monotonous', 'max'),
            std_monotonous=('monotonous', 'std'),
            avg_monotonous=('monotonous', 'mean')
        )

        out.to_csv(os.path.join(os.getcwd(), 'soundAQ.csv'), index = False)
        out_stats.to_csv(os.path.join(os.getcwd(), 'soundAQ_stats.csv'), index = True)

    if args.export in ["both", "event"]:
        event_dir = os.path.join(os.getcwd(), 'SoundAQnet_event_probability')
        print(f"load from {event_dir}")
        files = [p for p in Path(event_dir).iterdir() if p.is_file()]
        event_labels = ['Silence', 'Human_sounds', 'Wind', 'Water', 'Natural_sounds', 'Traffic',
                    'Sounds_of_things', 'Vehicle', 'Bird', 'Outside_rural_or_natural',
                    'Environment_and_background', 'Speech', 'Music', 'Noise', 'Animal']
        dic = {f"id":[], 'event_rank':[]}
        for i in tqdm(range(len(files)), total=len(files)):
            id = Path(files[i]).name.split('_event.txt')[0]
            df = pd.read_csv(files[i], header=None, names=["event"])
            prob = df["event"].to_list()
            reranked = sorted(zip(event_labels, prob), key=lambda x: x[1], reverse=True)
            dic['id'] += [id]
            dic['event_rank'] += [[label for rank, (label, score) in enumerate(reranked, 1)]]

        out = pd.DataFrame(dic)
        out.to_csv(os.path.join(os.getcwd(), 'soundEventRank.csv'), index = False)

if __name__ == "__main__":
   main()