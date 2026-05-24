# Slide 1

(video)

## Speech

> Hello everyone!  I’d like to start with a short video.

---

## Slide 2

Accent-sensitive speech recognition

Sergej Kurtasch

## Speech

> this is a funny example that shows the importance of correct pronunciation 
>
> There are several English apps that help improve pronunciation, but there is still room for a strong solution for German.

> So for my final project I focused on building and applying deep learning models for accent-sensitive speech recognition. 
>
> Let’s first take a look at how the system works.
>
(demo)
>
> So, what can we do here?
>
> We can select a sample sentence, press the record button, pronounce it, and the system will evaluate how accurate the pronunciation was.
>
> For example, I pronounce a sentence, press the button, and the system highlights which words or sounds were pronounced correctly and which were not.
>
> Providing this kind of feedback is relatively easy for someone who has just started learning German and is still not very confident in their pronunciation.
>
> But What happens if a native German speaker with perfect pronunciation uses this system?
>
> Will they get a one-hundred-percent result, with everything marked in green?
>
> Let’s check!
>
(demo) german long sentances
>
> And the uncomfortable truth is: not always. You see, the output is mach better than mein, but still some letters are red.
>
> And this is not because the native speaker’s pronunciation is bad.
>
> The issue lies in the limitations of the system itself.
>
> And this is exactly where the most challenging part of the project lies.
>
begins.

---

## Slide 3

WAV2Vec2 backen - packen

> The decision was to implement a second verification stage.
>
> At the first stage, the system performs full speech recognition, just as before.
>
> Then, only those audio segments where phonemes were flagged as incorrect are passed to the second verification stage. 
>
> This second stage works differently.

---

## Slide 4

На слайде картинка со спектрограммой и извлеченными из нее MFCC F1 F2 VOT


> Instead of analyzing the whole audio, we extract acoustic features and spectrograms from short segments corresponding to specific phonemes. These are then passed to a separate classifier. This model answers just one question: does the pronounced phoneme match the expected one?

---

## Slide 5

На слайде архитектура гибридной модели


> To start with I built 22 specialized models for 22 phoneme pairs. The best results came from a hybrid approach, combining acoustic features with spectrograms to maximize accuracy and other important scores.

---

## Slide 6

На слайде график с количеством ошибок по парам фонем до и после валидации


> This allowed to significantly reduce false mistaces. On the  slide, you’ll see how much we reduced errors compared to the baseline.
>
> This acoustic verification makes the feedback fairer and better justified. 

---

## Slide 7

Thank you for your attention.


> Thank you for your attention. I am now ready to answer any questions you may have. I would be happy for you to reach out to me later; you can find my contact details on this slide.
