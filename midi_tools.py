import mido
import torch

SUSTAIN_THRESHOLD = 64
TIME_SHIFT_INCREMENT_MS = 4

def midi_to_tensor(midi_path, max_length):
    """
    Converts a MIDI file into a tensor of MIDI instruction tokens.

    The tokens are encoded as follows:
        - 0 to 127: note_on (with the token value as the MIDI note number)
        - 128 to 255: note_off (`token - 128` gives the MIDI note number)
        - 256 to 287: velocity change (`token - 256` gives the velocity bucket; velocity is approximated as `bucket*4`)
        - 288 to 412: time shift tokens.
            - `(token - 288 + 1) * 8ms` is the time shift in ms.
            - A token of 412 represents a full second.
        - 413: Beginning of track
        - 414: Padding
    """

    def get_time_shift_tokens(time_shift) -> list:
        """Convert time shift to a list of prefered increments. Over 1 second shifts require multiple tokens."""
        ms_per_tick = 60_000_000 / (tempo * ppq)
        # Calculate the time shift in 8 ms increments
        total_timeshift_increments = round(time_shift * ms_per_tick / TIME_SHIFT_INCREMENT_MS)
        num_max_tokens = total_timeshift_increments // 125
        remainder = total_timeshift_increments % 125
        # if num_max_tokens > 0:
        #     print(num_max_tokens)
        if num_max_tokens == 0 and remainder == 0:
            return []
        elif remainder == 0:
            return [288 + 124] * num_max_tokens
        else:
            return [288 + 124] * num_max_tokens + [288 + remainder - 1]
        
    def pad_or_truncate_tensor(tensor, max_length):
        """Pad the tensor to the maximum length."""
        if len(tensor) < max_length:
            tensor += [414 for _ in (max_length - len(tensor))]
            return tensor
        elif len(tensor) > max_length:
            return tensor[:max_length]
        else:
            return tensor

    mid = mido.MidiFile(midi_path)
    ppq = mid.ticks_per_beat
    tempo = 0

    instruction_library = {
        'note_on': lambda x: x,         # 0-127
        'note_off': lambda x: 128 + x,  # 128-255
        'velocity': lambda x: 256 + x,  # 256-287
        'time_shift': lambda x: get_time_shift_tokens(x) # 288-412
    }

    midi_instructions = [413] # Beginning of track
    
    for track in mid.tracks:

        sustain = False
        sustained_notes = set()
        last_velocity = -1
        time_delta = 0

        for msg in track:

            # Set the tempo in the first instance of a tempo message
            if msg.is_meta:
                if (msg.type == "set_tempo") and (tempo == 0):
                    tempo = msg.tempo
                continue

            if msg.time > 0:
                time_delta += msg.time

            # Note On
            if (msg.type == 'note_on') and (msg.velocity != 0):

                # Time Shift
                if time_delta > 0:
                    midi_instructions.extend(instruction_library['time_shift'](time_delta))
                    time_delta = 0

                if sustain and (msg.note in sustained_notes):
                    midi_instructions.append(instruction_library['note_off'](msg.note))

                # Change velocity if there is a bucket change
                velocity_bucket = msg.velocity // 4 # 0-31
                if last_velocity != velocity_bucket:
                    midi_instructions.append(instruction_library['velocity'](velocity_bucket))
                    last_velocity = velocity_bucket
                    
                midi_instructions.append(instruction_library['note_on'](msg.note))

            # Note Off
            elif (msg.type == 'note_off') or ((msg.type == 'note_on') and (msg.velocity == 0)):

                # Time Shift
                if time_delta > 0:
                    midi_instructions.extend(instruction_library['time_shift'](time_delta))
                    time_delta = 0

                if sustain:
                    sustained_notes.add(msg.note)
                else:
                    midi_instructions.append(instruction_library['note_off'](msg.note))

            # Sustain Pedal
            elif (msg.type == 'control_change') and (msg.control == 64):

                if msg.value >= SUSTAIN_THRESHOLD: # Sustain pedal on                  
                    sustain = True
                elif sustain and (msg.value < SUSTAIN_THRESHOLD): # Sustain pedal off

                    # Time Shift
                    if time_delta > 0:
                        midi_instructions.extend(instruction_library['time_shift'](time_delta))
                        time_delta = 0

                    sustain = False
                    # Turn sustained_notes off
                    for sustained_note in sustained_notes:
                        midi_instructions.append(instruction_library['note_off'](sustained_note))
                    sustained_notes.clear()
                else:
                    continue

            else:
                continue

    # Pad or truncate the tensor to the maximum length 
    midi_instructions = pad_or_truncate_tensor(midi_instructions, max_length)

    return torch.LongTensor(midi_instructions), ppq, tempo
    
    
def tensor_to_midi(tensor, ppq, tempo, output_midi_path):
    """
    Converts a tensor of MIDI instruction tokens back into a MIDI file and saves it.

    The tokens are encoded as follows:
        - 0 to 127: note_on (with the token value as the MIDI note number)
        - 128 to 255: note_off (`token - 128` gives the MIDI note number)
        - 256 to 287: velocity change (`token - 256` gives the velocity bucket; velocity is approximated as `bucket*4`)
        - 288 to 412: time shift tokens.
            - `(token - 288 + 1) * 8ms` is the time shift in ms.
            - A token of 412 represents a full second.
    """

    # Convert the tensor to a list of tokens
    tokens = tensor.tolist() if torch.is_tensor(tensor) else tensor

    # Create a new MIDI file and track; set the PPQ from ppq
    mid = mido.MidiFile(ticks_per_beat=ppq)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    # Insert tempo meta message at the beginning of the track
    tempo_msg = mido.MetaMessage('set_tempo', tempo=tempo, time=0)
    track.append(tempo_msg)

    # Default velocity (if not updated by a velocity token)
    current_velocity = 64
    # Accumulate time (in ticks) from time shift tokens
    delta_ticks = 0

    # Pre-calculate ms_per_tick for conversion (ms per tick = 60000 / (tempo * ppq))
    ms_per_tick = 60_000_000 / (tempo * ppq)

    for token in tokens:
        # Time shift token (288-412)
        if 288 <= token <= 412:
            # Convert the time shift token back to (x) ms increments:
            increments = token - 288 + 1
            # Convert the number of (x)ms increments back to ticks:
            dt_ticks = int(round((increments * TIME_SHIFT_INCREMENT_MS) / ms_per_tick))
            delta_ticks += dt_ticks

        # Velocity token (256-287)
        elif 256 <= token <= 287:
            # Update current velocity (approximation)
            current_velocity = (token - 256) * 4

        # Note on token (0-127)
        elif 0 <= token <= 127:
            msg = mido.Message('note_on', note=token, velocity=current_velocity, time=delta_ticks)
            track.append(msg)
            delta_ticks = 0

        # Note off token (128-255)
        elif 128 <= token <= 255:
            note = token - 128
            msg = mido.Message('note_off', note=note, velocity=0, time=delta_ticks)
            track.append(msg)
            delta_ticks = 0

        else:
            # For any token outside the expected range, skip it
            continue

    # Save the MIDI file to the specified path
    mid.save(output_midi_path)
    return mid