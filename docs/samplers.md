# Which samplers work with SPEED

Tested on MiniMax H3: all 45 ComfyUI samplers, turbo LoRA, 8 steps, simple scheduler, BasicGuider, 4 stages, 0.3 MP, 2.3 s.

## Fine

`euler`, `euler_ancestral`, `euler_cfg_pp`, `euler_ancestral_cfg_pp`, `heun`, `heunpp2`, `exp_heun_2_x0`, `exp_heun_2_x0_sde`, `dpm_2`, `dpm_2_ancestral`, `dpmpp_2s_ancestral`, `dpmpp_2m`, `ddpm`, `lcm`, `ipndm`, `deis`, `cfgpp_ud10_ab`, `res_multistep`, `res_multistep_ancestral`, `gradient_estimation`, `er_sde`, `seeds_2`, `seeds_3`, `sa_solver`, `sa_solver_pece`, `ddim`.

`lms` and `ipndm_v` were clean here but smeared in an earlier 2-stage test, so check your output with them.

**SDE samplers** (`dpmpp_sde`, `dpmpp_2m_sde`, `dpmpp_3m_sde` and their `_gpu`/`_heun` versions) work, but give a different shot from Euler at the same seed. They replace almost all of the seed noise on their first step, so their own noise picks the composition. Native ComfyUI does the same.

## Broken

| Sampler | Why | Also broken without SPEED? |
|---|---|---|
| `dpm_fast`, `dpm_adaptive` | ComfyUI stops them one step before the end. On H3 that step is still 63 % noise. | Yes |
| `dpmpp_2m_cfg_pp`, `dpmpp_2s_ancestral_cfg_pp`, `res_multistep_cfg_pp`, `res_multistep_ancestral_cfg_pp`, `gradient_estimation_cfg_pp` | They need a negative prompt. BasicGuider has none, so they get zeros and blow up. They work with **CFGGuider** at cfg 1. | Yes |
| `uni_pc`, `uni_pc_bh2` | They skip any stage that is a single step. 4 stages at 8 steps gives two single-step stages. They work at **2 stages**. | No (they smear instead) |

**Rule of thumb:** give every stage at least 2 steps. More stages need more steps.
