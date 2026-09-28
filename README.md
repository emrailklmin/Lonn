<p align="center">
  <img src="assets/Lonnlogo.png" width="300" alt="Lönn AI">
</p>


AI infrastructure for intelligent LLM routing, cost optimisation and observability.

For 

## Setup

1. Copy `.env.example` to `.env`.
2. Fill in your NVIDIA NIM API key (get one at https://build.nvidia.com) as
   `NVIDIA_NIM_API_KEY=nvapi-...` — that exact variable name is required, it's
   what LiteLLM's `nvidia_nim/...` provider reads.
3. `.env` is gitignored — never commit it.
