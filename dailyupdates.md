# Day 1

Project Creation


# Day 2

## Testing
So, I began testing different models for TTS: Gemini, Qwen 3 TTS. Gemini is very fast, however, free api rate limits are very strict, and you use them up very fastly, however, qwen 3 tts is slower, but actually sound great, which is why I will leave it like that.

## Creations

I have been developing a new script that combines current functions, as I need the raspberry pi and the actual robot, right now I cannot do pretty much hardware, so I should focus on software. So basically it uses yolov11, gemini api to use gemini flash lite latest to analyze images and qwen tts. So, if you ask something like "Tell me a short story", it wont use the gemini API, just the gpt-oss-120b and qwen tts, however, if you tell it "What am I holding", it will pass an image of current frame when action analyze environment is recieved and the AI will analyze it and give back a response which I think is just great.

## Next Steps

I would basically just try and maximize efficiency for the TTS speech to feel like no-latency and actually like a real conversation, we could divide output into parts, first sentence, very divided and tehn just generate audio by chunks with a very efficient multiprocessing generation

I would also start actually making code to "test" how gpt-oss-120b would "follow instructions" to control "the robot" for example, if I say go forward for 3 seconds, then turn left and go forward 2 seconds it should output: 

[output: Movement, action: 3secF,TL(90 degree),2secF] 

or something similar. I could create a virtual environment with a 5x5 with boxes and create a "virtual robot" simulating the actual robot, where it requires the format described to function, then just input instructions by text and analyze how efficient it is.
