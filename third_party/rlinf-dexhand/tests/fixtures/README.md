These are actual ADC captures from the two connected gloves on 2026-09-11.
They are inputs, not inferred vendor identities or ground-truth hand poses.
PSI1 has 21 channels; PSI2 has 22. The hex frame is a CRC-valid PSI2 response.
The numerical tests additionally use deterministic seeded sequences and compare
against the original mapper/skeleton/optimizer, rather than comparing a function
against its own saved output. Replay refreshes delivery timestamps.
