import jax
from jax import random, numpy as jp

from madrona_mjx import BatchRenderer 
from madrona_mjx.viz import VisualizerGPUState, Visualizer

import argparse

from mjx_env import MJXEnvAndPolicy

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--window-width', type=int, required=True)
arg_parser.add_argument('--window-height', type=int, required=True)
arg_parser.add_argument('--batch-render-view-width', type=int, required=True)
arg_parser.add_argument('--batch-render-view-height', type=int, required=True)

args = arg_parser.parse_args()

viz_gpu_state = VisualizerGPUState(args.window_width, args.window_height, args.gpu_id)

mjx_wrapper = MJXEnvAndPolicy.create(random.key(0), args.num_worlds)

renderer = BatchRenderer(
    mjx_wrapper.env, mjx_wrapper.mjx_state, args.gpu_id, args.num_worlds,
    args.batch_render_view_width, args.batch_render_view_height,
    False, viz_gpu_state.get_gpu_handles())

def step_fn(mjx_wrapper):
  mjx_wrapper = mjx_wrapper.step()

  # Note that the renderer prim is effectful so it won't get optimized out
  # even if not used
  renderer.render(mjx_wrapper.mjx_state)
  return mjx_wrapper

#step_fn = jax.jit(step_fn)
#step_fn = step_fn.lower(mjx_wrapper)
#step_fn = step_fn.compile()

visualizer = Visualizer(viz_gpu_state, renderer.madrona)
visualizer.loop(renderer.madrona, step_fn, mjx_wrapper)
