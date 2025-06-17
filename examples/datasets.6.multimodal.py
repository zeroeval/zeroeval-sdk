import os
import sys
from pathlib import Path
import zeroeval as ze

# -----------------------------------------------------------------------------
# README
# -----------------------------------------------------------------------------
# This dataset is designed to benchmark speech-to-text (ASR) models.
# It takes the sample WAV files located in the sibling directory `sample_audio/`
# and builds a ZeroEval multimodal dataset where each record contains:
#   • segment_id           – Full segment identifier from the file name (e.g. "Speaker0050_000")
#   • language             – Spoken language (assumed English)
#   • expected_transcript  – Reference transcript (left blank for now – to be
#                            filled in later once ground-truth is available)
#   • audio_clip           – The actual audio file added via `add_audio()`
#
# When reference transcripts become available you can iterate over the dataset
# and update the `expected_transcript` field so that experiments can evaluate
# WER/CER metrics.
# -----------------------------------------------------------------------------

ze.init(api_key="sk_ze_r8Mf8-nXn-llv4kJIdxBwRagcQbQxjFDm6_4sai8xrU")

# Locate the directory that contains the `.wav` files
current_dir = Path(__file__).parent.absolute()
sample_audio_dir = current_dir / "sample_audio"

if not sample_audio_dir.exists():
    print(f"Error: Sample audio directory not found at {sample_audio_dir}")
    print("Please create this directory and add WAV files before running this script.")
    sys.exit(1)

# Collect all WAV files (sorted for reproducibility)
audio_files = sorted(sample_audio_dir.glob("*.wav"))

if len(audio_files) == 0:
    print(f"Error: No .wav files found in {sample_audio_dir}")
    sys.exit(1)

# Define reference transcripts for each audio file
reference_transcripts = {
    "Speaker0050_000": "That morning, when Michael Hennessey's girl Mary—a girl sixteen years old—carried the can of milk to the rear door of the silent house, she was nearly a quarter of an hour later than usual, and looked forward to being soundly raided. \"He's up and waiting for it,\" she said to herself, observing the scullery door ajar. \"Won't I catch it? It's him for growling and snapping at a body, and it's me for always being before or behind time. Bad luck to me. There's no place in him.\" Mary pushed back the door and passed through the kitchen, serving herself all the while to meet the obstrucations which she supposed were lying in wait for her. The sunshine was blinding without, but sifted through the green jalousies it made a grey, crepuscular light within. As the girl approached the table, on which a plate with a knife and fork had been laid for breakfast, she noticed, somewhat indistinctly at first, that there was no one in the kitchen.",
    "Speaker0050_001": "first, a thin red line running obliquely across the floor from the direction of the sitting-room, and ending near the stove, where it had formed a small pool. Mary stopped short, scarcely conscious why, and peered instinctively into the adjoining apartment. Then, with a smothered cry, she let fall the milk-can, and a dozen white rivulets, in strange contrast to that one dark red line which first startled her, went meandering over the kitchen floor. With her eyes riveted upon some object in the next room, the girl retreated backward, slowly and heavily dragging one foot after the other, until she reached the gallery door. Then she turned swiftly and plunged into the street. Twenty minutes later, every man, woman, and child in Stillwater knew that Mr. Shackford had been murdered. Mary Hennessey had to tell her story a hundred times during the morning, for each minute brought to Michael's tenement a fresh listener, hungry for the details at first hand. How was it, Molly? Tell about it, dear. Don't be a-",
    "Speaker0050_002": "asking me!\" cried Molly, pressing her palms to her eyes as if to shut out the sight, but taking all the while a secret creepy satisfaction in living the scene over again. It was kinder dark in the other room, and there he was, lying in his night-gown, with his face turned towards me so, looking mighty severe-like, just as if he was a-going to say, \"It's late with the milk he are, you hussy. Away he had a-spakin'.\" But he didn't spake, Molly darlin'. \"Nither a word. He was stoned-dead, don't you see. It was that still. You could hear me heart-beat, saven there wasn't a drop o' beat in it. I let go the can, sure. Then I backed out, with me eye on him all the while, afear to death that he would up and spake them words. The poor child! For the likes o' her to be wakin' up the mirthed man in the mornin'!\" There was little or no work done that day in Stillwater, outside the mills, and they were not running full-handed. A number of men from the Meantawona Ironworks and Slocum's Yard—Slocum employed some seventy or eighty hands—lounged about the streets in their",
    "Speaker0050_003": "blouses, or stood in knots in front of the tavern, smoking short clay pipes. Not an urchin put in an appearance at the small red-brick building on the turnpike. Mr. Pinkham, the schoolmaster, waited an hour for the recusants, then turned the key in the lock and went home. Dragged-looking women, with dishcloth or dustpan in hand, stood in doorways or leaned from windows, talking in subdued voices with neighbors on the curbstone. In a hundred faraway cities the news of the suburban tragedy had already been read and forgotten, but here the horror stayed. There was a constantly changing crowd gathered in front of the house in Welch's Court. An inquest was being held in the room adjoining the kitchen. The court, which ended at the gate of the cottage, was fringed for several yards on each side by rows of squalid, wandering children, who understood it, that Coroner Whidden was literally to sit on the dead body. Mr. Whidden, a limp, inoffensive little man who would not have dared to sit down on a fly. He had passed, pallid and perspiring.",
    "Speaker0050_004": "to the scene of his perfunctory duties. The result of the investigation was awaited with feverish impatience by the people outside. Mr. Shackford had not been a popular man. He had been a hard, avaricious, passionate man, holding his own way remorselessly. He had been the reverse of popular, but he had long been a prominent character in Stillwater, because of his wealth, his endless lawsuits, and his eccentricity, an illustration of which was his persistence in living entirely alone in the isolated and dreary old house that was henceforth to be inhabited by his shadow. Not his shadow alone, however, for it was now remembered that the premises were already held in fee by another phantasmal tenant. At a period long anterior to this, one Lydia Sloper, a widow, had died an unexplained death under the same roof. The coincidence struck deeply into the imaginative portion of Stillwater. The widow Sloper and old Shackford have made a match of it, remarked a local humorist, in a grimmer vein than customary. Two ghosts had now",
    "Speaker0050_005": "set up housekeeping, as it were, in the stricken mansion, and what might not be looked for in the way of spectral progeny. It appeared to the crowd in the lane that the jury were unconscionably long in arriving at a decision, and when the decision was at length reached it gave but moderate satisfaction. After a spendthrift waste of judicial mind, the jury had decided that the death of Lemuel Shackford was caused by a blow on the left temple, inflicted with some instrument not discoverable in the hands of some person or persons unknown. We knew that before, grumbled a voice in the crowd, when, to relieve public suspense, Lawyer Perkins, a long, lank man with stringy black hair, announced the verdict from the doorstep. The theory of suicide had obtained momentary credence early in the morning, and one or two still clung to it with the tenacity that characterizes persons who entertain few ideas. To accept this theory, it was necessary to believe that Mr. Shackford had ingeniously hidden the weapon after striking himself dead with a single blow.",
    "Speaker0050_006": "No, it was not suicide. So far from intending to take his own life, Mr. Shackford, it appeared, had made rather careful preparations to live that day. The breakfast-table had been laid overnight, the coals left ready for kindling in the Franklin stove, and a kettle, filled with water to be heated for his tea or coffee, stood on the hearth. Two facts had sharply demonstrated themselves. First, that Mr. Shackford had been murdered, and second, that the spur to the crime had been the possession of a sum of money, which the deceased was supposed to keep in a strong-box in his bedroom. The padlock had been wrenched open, and the less valuable contents of the chest, chiefly papers, scattered over the carpet. A memorandum among the papers seemed to specify the respective sums in notes and gold that had been deposited in the box. A document of some kind had been torn into minute pieces and thrown into the waste-basket. On close scrutiny, a word or two here and there revealed the fact that the document was of a legal character. The fragments were put into an envelope, and given in charge",
    "Speaker0050_007": "of Mr. Shackford's lawyer, who placed seals on that and on the drawers of an escritoire, which stood in the corner and contained other manuscript. The instrument with which the fatal blow had been dealt, for the autopsy showed that there had been but one blow, was not only not discoverable, but the fashion of it defied conjecture. The shape of the wound did not indicate the use of any implement known to the jurors, several of whom were skilled machinists. The wound was an inch and three-quarters in length, and very deep at the extremities. In the middle it scarcely penetrated to the cranium. So peculiar a cut could not have been produced with the claw part of a hammer, because the claw is always curved, and the incision was straight. A flat claw, which is used in opening packing cases, was suggested. A collection of several sizes manufactured was procured, but none corresponded with the wound. They were either too wide or too narrow. Moreover, the cut was as thin as the blade of a case-knife. That was never done by any tool in these parts,\" declared Stevens, the foreman of the finishing",
    "Speaker0050_008": "shop at Slocum's. The assassin or assassins had entered by the scullery door, the simple fastening of which, a hook and staple, had been broken. There were footprints in the soft clay path leading from the side gate to the stone step, but Mary Hennessy had so confused and obliterated the outlines that now it was impossible accurately to measure them. A half-burnt match was found under the sink, evidently thrown there by the burglars. It was of a kind known as the safety match, which can be ignited only by friction on a strip of chemically prepared paper glued to the box. As no box of this description was discovered, and as all the other matches in the house were of a different make, the charred splinter was preserved. The most minute examination failed to show more than this. The last time Mr. Shackford had been seen alive was at six o'clock the previous evening. Who had done the deed? Tramps, answered Stillwater with one voice, though Stillwater lay somewhat out of the natural highway, and the Tramp, that bitter blossom of civilization,",
    "Speaker0050_009": "whose seed was blown to us from over the seas, was not then so common by the New England roadsides as he became five or six years later. But it was intolerable not to have a theory, it was that or none, for conjecture turned to no one in the village. To be sure, Mr. Shackford had been in litigation with several of the corporations, and had had legal quarrels with more than one of his neighbours. But Mr. Shackford had never been victorious in any of these contests, and the incentive of revenge was wanting to explain the crime. Besides, it was so clearly robbery. Though the gathering around the Shackford house had reduced itself to half a dozen idlers, and the less frequented streets had resumed their normal aspect of dullness, there was a strange electric quality in the atmosphere. The community was in that state of suppressed agitation and suspicion which no word adequately describes. The slightest circumstance would have swayed it to the belief in any man's guilt, and, indeed, there were men in Stillwater quite capable of disposing a fellow-creature for a much smaller price."
}

records = []
for wav_path in audio_files:
    # Example file name: Speaker0050_003.wav
    stem = wav_path.stem  # -> "Speaker0050_003"
    
    # Get the reference transcript for this specific file
    expected_transcript = reference_transcripts.get(stem, "")

    record = {
        "segment_id": stem,  # Use full segment ID (e.g. "Speaker0050_000")
        "language": "English",
        "expected_transcript": expected_transcript
    }
    records.append(record)

# Create the dataset in ZeroEval
asr_dataset = ze.Dataset(
    name="ReadAloudStoryAudio",
    description=(
        "Audio clips of a single speaker reading a story out loud. "
        "Each record is intended for automatic speech recognition (ASR) evaluation."
    ),
    data=records,
)

# Attach the actual audio to each row under the column name `audio_clip`
for idx, wav_path in enumerate(audio_files):
    asr_dataset.add_audio(
        row_index=idx,
        column_name="audio_clip",
        audio_path=str(wav_path)
    )
    print(f"Attached audio file {wav_path.name} to row {idx}.")

print("\nDataset structure:")
print(f"Name: {asr_dataset.name}")
print(f"Description: {asr_dataset.description}")
print(f"Number of records: {len(asr_dataset)}")
print(f"Columns: {asr_dataset.columns}")

# Uncomment the following line to push the dataset to your ZeroEval workspace
asr_dataset.push() 