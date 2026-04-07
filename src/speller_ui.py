import time
try:
    from psychopy import visual, core, event
except ImportError:
    print("PsychoPy not installed. This is a placeholder for the Stimulus Interface.")
    visual = None

def run_speller_ui():
    """
    Minimal PsychoPy P300 Speller UI.
    Requirement: speller_ui.py (Gap #3)
    """
    if visual is None:
        print("Please install psychopy to run the UI: pip install psychopy")
        return

    win = visual.Window([800, 600], monitor="testMonitor", units="deg", fullscr=False)
    
    # 6x6 Grid of characters
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    stimuli = []
    for i, char in enumerate(chars):
        row = i // 6
        col = i % 6
        stim = visual.TextStim(win, text=char, pos=(col*2 - 5, 5 - row*2))
        stimuli.append(stim)

    print("Starting Speller UI... Press 'q' to quit.")
    
    while True:
        for stim in stimuli:
            stim.draw()
        win.flip()
        
        # Simulate a Row/Column flash every 175ms (SOA)
        time.sleep(0.175)
        
        if 'q' in event.getKeys():
            break

    win.close()
    core.quit()

if __name__ == "__main__":
    run_speller_ui()
