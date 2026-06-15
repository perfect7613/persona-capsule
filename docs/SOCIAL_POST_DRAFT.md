# Persona Capsule Launch Thread

Post the numbered text as a single X thread. Add the suggested media before
publishing each post. Replace `[DEMO_URL]` after uploading the final video.

## 1/15

What if the way you communicate could become a portable AI object?

For the @huggingface + @Gradio #BuildSmallHackathon, we built Persona Capsule:
approved messages become a private, live-steered MiniCPM personality, anime
card, optional voice and shareable chat.

Thread:

**Media:** 8-12 second hero montage: messages -> steering comparison -> anime
card -> public capsule.

## 2/15

The spark was @dlouapre's @huggingface video, "Steering LLM Behavior Without
Fine-Tuning."

I'm increasingly interested in mechanistic interpretability. It made me ask:
could activation steering become a product ordinary people can actually feel?

https://youtu.be/F2jd5WuT-zg

**Media:** Short screen capture of the video title, followed by Persona
Capsule's steering screen. Credit the original video on-screen.

## 3/15

Most AI "personality" apps are a system prompt.

Persona Capsule changes MiniCPM's internal activations while it generates. The
weights stay unchanged. The intervention exists only for that request, then its
hooks are removed.

Same model. Different internal direction.

**Media:** Simple animation showing `prompt -> MiniCPM layers -> answer`, then a
temporary steering arrow entering the middle layers.

## 4/15

The core math is contrastive activation:

1. Take writing examples the user approved.
2. Create neutral versions with the same meaning.
3. Measure both inside selected MiniCPM layers.
4. Average `style - neutral`.

That difference becomes the temporary personality direction.

**Media:** Formula card:
`v_l = mean_i(mean_t(h_l(style_i)) - mean_t(h_l(neutral_i)))`.

## 5/15

During generation we apply:

`h'_l = h_l + alpha * v_l`

`alpha` is the strength control. Too little is invisible; too much can damage
coherence, so we normalize and cap the intervention. We apply it across several
middle layers, where abstract features are more likely to live.

**Media:** Steering-strength slider moving from baseline to recognizable style,
then stopping before an exaggerated output.

## 6/15

The test is deliberately visible: one prompt goes through the same
@OpenBMB MiniCPM4.1-8B twice.

One answer is baseline. One receives live activation steering.

You can compare them side by side instead of trusting a hidden personalization
claim.

**Media:** 15-20 second baseline-versus-steered Shakespeare example. Highlight
the phrases that changed.

## 7/15

@modal runs the GPU-heavy parts as separate runtimes:

- MiniCPM activation extraction + generation
- FLUX image generation
- VoxCPM2 speech
- Nemotron evaluation

The @huggingface Space stays responsive while each open-weight model gets the
hardware it needs.

**Media:** Architecture animation with the Gradio Space dispatching four jobs
to Modal.

## 8/15

A capsule should feel like an object, not a settings file.

@bfl_ai FLUX.2 Klein 4B + an anime LoRA turns the profile into a fictional
collectible card. Descriptors influence expression, posture, palette and
motifs; private messages never go to the image model.

**Media:** Anime card generation reveal and final 1200x628 social card.

## 9/15

For voice, we replaced a closed cloning dependency with @OpenBMB VoxCPM2.

Voice is optional, consent-gated and private. References have a retention
window, samples are shared separately, and failed cleanup stays visible and
retryable instead of being silently ignored.

**Media:** 10-15 second consent screen -> generate voice -> audio playback clip.

## 10/15

We also built capsule fusion and a blinded battle.

Two compatible steering recipes can be mixed by weight. Then MiniCPM produces
anonymous candidates and @NVIDIAAI Nemotron 3 Nano judges both A/B and B/A
orders to reduce position bias.

**Media:** Fusion slider followed by the two-order Nemotron battle result.

## 11/15

Privacy shaped the architecture:

- private by default
- redact before analysis
- retain only approved examples
- store the recipe, not a permanent activation tensor
- separate private records from public projections
- let creators unpublish without deleting their capsule

**Media:** Five-second privacy checklist animation, followed by the public
preview controls.

## 12/15

Publishing creates more than an image.

Every public capsule has a stable page where friends can chat with its
live-steered MiniCPM and play "Do you really know me?" Two answers appear; one
is baseline and one is steered. Can they recognize the creator's signal?

**Media:** Public link opening in a private browser, one chat response, then the
challenge reveal.

## 13/15

@OpenAIDevs Codex was a real engineering collaborator, not a one-prompt code
generator: plan audit, PRD, GitHub issues, architecture, implementation, tests,
UI inspection, debugging and deployment.

Codex-authored commits are publicly attributed in Git history. @reach_vb

**Media:** Fast scroll through the PRD, issue slices, tests and attributed Git
commits.

## 14/15

The messy parts taught us the most: zero-magnitude vectors, weak contrast pairs,
language drift, over-steering, OAuth redirects and public files disappearing
after Space rebuilds.

Building the demo pushed activation steering from a neat experiment into a
complete product system.

**Media:** Quick "failed -> learned -> fixed" montage using real, non-sensitive
error fragments.

## 15/15

Try Persona Capsule and tell me whether the steered answer actually feels
recognizable:

Space: https://huggingface.co/spaces/build-small-hackathon/persona-capsule

Code: https://github.com/perfect7613/persona-capsule

Demo: [DEMO_URL]

Built small, open-weight and inspectable.

**Media:** Final card with the Space URL and "Do you really know me?"

## Thank-You Reply

Post these as replies after the thread so the launch posts stay readable.

Thank you to the hosts and partner teams who made #BuildSmallHackathon possible:
@huggingface @Gradio @OpenBMB @OpenAIDevs @NVIDIAAI @modal @bfl_ai @jetbrains
@Cohere_Labs.

And thank you to the people whose tools, teaching and community energy helped
make the event: @ben_burtenshaw @yvrjsharma @_akhaliq @abidlabs @dlouapre and
@reach_vb, with @Thom_Wolf and @ClementDelangue.

## Verified X Handles

Checked on June 15, 2026:

| Organization or person | X handle |
| --- | --- |
| Hugging Face | `@huggingface` |
| Gradio | `@Gradio` |
| OpenBMB / MiniCPM / VoxCPM2 | `@OpenBMB` |
| OpenAI Developers / Codex | `@OpenAIDevs` |
| NVIDIA AI / Nemotron | `@NVIDIAAI` |
| Modal | `@modal` |
| Black Forest Labs / FLUX | `@bfl_ai` |
| JetBrains | `@jetbrains` |
| Cohere Labs | `@Cohere_Labs` |
| Ben Burtenshaw | `@ben_burtenshaw` |
| Yuvi / Yuvraj Sharma | `@yvrjsharma` |
| AK | `@_akhaliq` |
| Abubakar Abid | `@abidlabs` |
| David Louapre | `@dlouapre` |
| Vaibhav "VB" Srivastav | `@reach_vb` |
| Thomas Wolf | `@Thom_Wolf` |
| Clem Delangue | `@ClementDelangue` |
