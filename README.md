# noop-plus

# Types of permutations

We use 3 paraphrase types to permute the given datasets. The original NoOp permutation tackles the paraphrase type "addition".
## Paraphrase Type: Addition
Add irrelevant context, including references to the dataset, using prompting.
## Paraphrase Type: Lexicon-Changes
Exchange words with second highest logprobs
## Paraphrase Type: Syntax-Changes
Switch up the syntax of individual sentences using GPT-4o-mini.
## (Paraphrase Type: Deletion)
Remove irrelevant context using prompting. (Might not always be applicable)