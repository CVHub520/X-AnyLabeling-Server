"""GroundAnything image-token preparation for the plain-Qwen3 DLM reference server."""

NON_THINKING_PREFIX = '<think>\n\n</think>\n\n'
IMAGE_TOKEN = '<|image_pad|>'


def prepare_qwen3_inputs(processor, messages):
    images = []
    for message in messages:
        content = message.get('content')
        if isinstance(content, str):
            continue
        if not isinstance(content, list) or any(
            not isinstance(item, dict) for item in content
        ):
            raise ValueError(
                'message content must be text or a list of content objects'
            )
        images.extend(
            item['image'] for item in content if item.get('type') == 'image'
        )
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not prompt.endswith(NON_THINKING_PREFIX):
        prompt += NON_THINKING_PREFIX
    image_inputs = (
        processor.image_processor(images=images, return_tensors='pt')
        if images
        else {}
    )
    grids = image_inputs.get('image_grid_thw', [])
    merge = int(processor.image_processor.merge_size) ** 2
    sizes = [int(grid.prod()) for grid in grids]
    if any(size <= 0 or size % merge for size in sizes):
        raise ValueError('invalid GroundAnything image grid')
    counts = [size // merge for size in sizes]
    pieces = prompt.split(IMAGE_TOKEN)
    if len(pieces) != len(counts) + 1:
        raise ValueError(
            'GroundAnything image placeholder count differs from processed images'
        )
    prompt = pieces[0] + ''.join(
        IMAGE_TOKEN * count + tail for count, tail in zip(counts, pieces[1:])
    )
    text_inputs = processor.tokenizer(
        [prompt], return_tensors='pt', padding=False
    )
    return {**dict(text_inputs), **dict(image_inputs)}
