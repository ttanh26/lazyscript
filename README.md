# Introduction
Generally, voice speed would be faster than writing speed. Therefore, it seems hard for audiences to write the full details of what they heard in a meeting, conference. They often will make note by keywords or main idea instead, but it will make their notes have less information and be useless if they need a summary. Some people (include me), often make an audio record to hear again later. By doing that, we can make sure that we would not miss any details in the conference. But, hearing again would take a lot of time. Applying Deep Learning models for making transcript automatically would save time and increase efficiency for us.

# Project goals
Try to classify different speaker in a conference and what did they say.

# Approach
Different speaker will have different ways of speaking that makes their voice become special. Therefore, based on some features of an audio signal like amplitudes and frequencies, we could detect the changing point between speakers. From these point, we could split the original one into smaller files which contains 1 speaker only in each file. Finally, by applying the Google API for each small files, we could get the transcript and then concatenate them to a long transcript for the whole audio files in the order of time.

# Demo

