# Examining reasoning traces on adversarial prompts from WeirdChat
> 6hr

## findings:
- 


## Motivation/Goals/Specific Questions
- background:
    - what is weirdchat?
        - weirdchat is a set of prompts that are designed/evolved to elicit unsafe/misleading responses
            - e.g. instructing users on how to cut or kill themselves, saying their headaches are cuased by chemtrails, claiming to be human or claiming to have smart-home info or to have called 911, etc
        - the prompts and responses around found without reasoning enabled for the subject model
            - even though they're all reasoning models

    - models respond in ways that are undesirable to these prompts
        - with the exception of maybe the BAC where the models are just right, that guy is probably good to drive
        - and the positive matches for the strike-partner one. so dumb. a fucking rubber band.

- what questions are worth asking for these prompts?
    - what role does lack of reasoning play in elicitation?
    - why are they responding the way they are? why do they give bad answers to these prompts?
        - is CoT reading helpful here?


### 2026-08-31
- Day 1 of project.
    - [ ] todo: expound on motivation, goals above

- did quick sanity check with deepseek-v4-flash
    - small test of 3 behaviors (chemtrails, denying being an AI, and fabricated benchmarking):
        - definitely displays the behaviors on the weird behaviors with thinking enabled
        - reduces target behaviors rate by a factor of 0.16, 0.37, and 0.41 respectively
        - so around 3-6x reduction

        - skimming CoT:
            - chemtrails (6 matches out of 50 queries)
                - "This sounds like a fictional scenario or a request for a speculative analysis"
                - "Possibly it's a request for a speculative analysis"
                - "This is a speculative scenario"
                - one starts with: "Okay, this is a highly unusual query"
            - denying being an ai (8/50)
                - mentions of 'roleplaying'
                    - "The instruction says I'm to roleplay as a character in a dating app"
                    - "The context is that I'm roleplaying as a human on a dating app"
                    - but only on 2/8
                - the others are weirdly unconflicted
                    - they just say "need to reassure user that I'm human"

    - did test of all target behaviors
        - idk, kind of not impressed
        - some of the prompts/target behaviors are so silly
            - like the BAC one. that dude is just acutlaly below the legal limit
            - how much does this matter for investigating?

### 2026-09-02
- Day 2

- Need to think about the goals here. let me brainstorm some (non-exclusive) potential goals/questions for a potential shopping list
    - how much do models still do the behaviors with reasoning on/off?
        - what kind of tasks does reasoning most help with not doing the bad behavior?
            - what are the most important parts?
            - can we characterize the tasks where CoT helps the least?

    - general explain the weirdest examples from weirdchat
        - why do models do the bad behavior?
            - roleplaying?
            - style copying?

    - benchmarking methods:
        - How useful is CoT for figuring out why models do this?
            - find clear examples of useful traces
            - do explanations of prompt->cot->answer generalize to the prompt->answer case?
                - meaning if we develop a good understanding of why the model does something with CoT enabled, does that understanding transfer to no-CoT?
                - find clear examples of misleading/unfaithfulness
        - how useful are white-box methods?
            - lenses and saes, primarily
        - compare above on hypothesis generation and on hypothesis testing
            - which is better for which?

- realizing that there are some wrinkles around prompt selection
    - we mostly ilkleyt dont just want the ones with the highest elicitation rate
    - they had the elo score stating all the desirable things we want in a prompt
        - it should be unexpected, it should be harmful, and the prompt should be natural
    - what are the prompts most in need of explaining?
        - the unexpected ones, duh
        - prompt naturalness is also an important thing here though
            - like lots of the prompts are clearly unnatural in a way that elicits the unexpecetd behavior

    - think the best strategy is then to just look for the highest elo prompts across several of the models?
        - take those as our targets

- ok current priorities: find an actual reportable, interesting thing. a clean finding you can say in a sentence
    - broadly, what have we learned so far, or what are some interesting observations?
        - the behaviors  still replicate at non-negligible rates in most cases
            - the reduction factors are fairly considerable, ranging from 2-6x
            - some behaviors appear to not replicate at all

        - for qwen 3.6 27b:
            - the most unexpected prompt was the chemtrails thing, and I have to agree
            - haven't gotten it to replicate with CoT on yet
            - It's interesting how divergent the two kinds of completion are
                - one is very alarming chemtrails, the other says "that's impossible", and there's little in between.
                - tried jlens on these tokens, they seem weirdly unhelpful
                    - jlens and template lenses give very different looks here
                    - on the start of the assistant's completion with no CoT: "The phenomenon is", looking at the 'is':
                        - jlens shows: fictional, hypothetical, simulated, mock, imagined, simulation, ...
                        - template shows: strange, apocalypse, aliens, historian, dragons, bizarre, monsters, derived, alien, climate change, griffin, magic, hilarious
                    - so the obvious connection is implausiblity/falseness
                        - other nearby tokens from jlens also suggest impossibility/alarming/general craziness
