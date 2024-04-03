def limit_prompt(prompt, max_size):
    # Split the prompt into context and question
    context, question = prompt['context'], prompt['question']

    # Check if the context exceeds the maximum size
    if len(context) > max_size:
        # Truncate the context while preserving question
        limited_context = context[:max_size]
        # Combine limited context and question into a new prompt
        limited_prompt = {'context': limited_context, 'question': question}
        return limited_prompt
    else:
        return prompt

def condense_prompt(prompt: ChatPromptTemplate, llm, max_tokens: int) -> ChatPromptTemplate:
    messages = prompt.to_messages()
    print(messages)
    num_tokens = llm.get_num_tokens_from_messages(messages)
    print(f'number of tokens in prompt: {num_tokens}')
    ai_function_messages = messages[2:]
    print(f'ai_function messages: {ai_function_messages}')
    while num_tokens > max_tokens:
        ai_function_messages = ai_function_messages[2:]
        num_tokens = llm.get_num_tokens_from_messages(
            messages[:2] + ai_function_messages
        )
    messages = messages[:2] + ai_function_messages
    return ChatPromptTemplate(messages=messages)