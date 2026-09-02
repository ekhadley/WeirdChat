# Examining reasoning traces on adversarial prompts from WeirdChat
> 5hr

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
                - mentions of 'rolaplaying'
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
        - it should be unexpected, it should be harmful, 
