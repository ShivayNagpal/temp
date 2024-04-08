import { NextApiRequest, NextApiResponse } from 'next';
import { getServerSession } from 'next-auth';

import { DEFAULT_SYSTEM_PROMPT } from '@/utils/app/const';
import { OpenAIStream } from '@/utils/server';
import { ensureHasValidSession, getUserHash } from '@/utils/server/auth';
import { getErrorResponseBody } from '@/utils/server/error';
import { saveLlmUsage, verifyUserLlmUsage } from '@/utils/server/llmUsage';
import { createMessagesToSend } from '@/utils/server/message';
import { getTiktokenEncoding } from '@/utils/server/tiktoken';

import { ChatBodySchema } from '@/types/chat';
import { OpenAIModelID, OpenAIModelType } from '@/types/openai';

import { authOptions } from '@/pages/api/auth/[...nextauth]';

import { OpenAIEmbeddings } from '@langchain/openai';
import { QdrantClient } from '@qdrant/js-client-rest';
import loggerFn from 'pino';

const logger = loggerFn({ name: 'chat' });

const handler = async (req: NextApiRequest, res: NextApiResponse) => {
  if (!(await ensureHasValidSession(req, res))) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  const session = await getServerSession(req, res, authOptions);
  if (session && process.env.AUDIT_LOG_ENABLED === 'true') {
    logger.info({ event: 'chat', user: session.user });
  }

  const userId = await getUserHash(req, res);
  const { model, messages, key, prompt, temperature } = ChatBodySchema.parse(
    req.body,
  );

  try {
    await verifyUserLlmUsage(userId, model.id);
  } catch (e: any) {
    return res.status(429).json({ error: e.message });
  }

  const encoding = await getTiktokenEncoding(model.id);
  try {
    let systemPromptToSend = prompt;
    if (!systemPromptToSend) {
      systemPromptToSend = DEFAULT_SYSTEM_PROMPT;
    }
    let {
      messages: messagesToSend,
      maxToken,
      tokenCount,
    } = createMessagesToSend(
      encoding,
      model,
      systemPromptToSend,
      1000,
      messages,
    );

    if (messagesToSend.length === 0) {
      console.error('No messages to send');
    }

    const embeddings = new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY,
    });

    const createVectorEmbedding = await embeddings.embedQuery(
      messagesToSend[0].content,
    );

    const client = new QdrantClient({
      url: process.env.QDRANTCLIENT_URL,
      apiKey: process.env.QDRANTCLIENT_API_KEY,
    });

    const searchResult = await client.search('norrsken-funds', {
      vector: createVectorEmbedding,
      limit: 8,
    });

    const commonPrompt: {
      content: string;
      role: 'system' | 'assistant' | 'user';
    } = {
      role: 'system',
      content: `The user is asking about ${
        messagesToSend[messagesToSend.length - 1].content
      }.
        Here are some related articles: from this ${JSON.stringify(
          searchResult,
        )}
        Provie the user with the information they need.`,
    };

    if (searchResult.length > 1) {
      messagesToSend[messagesToSend.length - 1] = commonPrompt;
    } else {
      messagesToSend = [commonPrompt];
    }

    const stream = await OpenAIStream(
      {
        id: OpenAIModelID.GPT_4_TURBO_PREVIEW,
        type: OpenAIModelType.CHAT,
        name: 'GPT-4 Turbo',
        maxLength: 4096,
        tokenLimit: 128000,
      },
      systemPromptToSend,
      temperature,
      key,
      messagesToSend,
      maxToken,
    );
    res.status(200);
    res.writeHead(200, {
      Connection: 'keep-alive',
      'Content-Encoding': 'none',
      'Transfer-Encoding': 'chunked',
      'Cache-Control': 'no-cache',
      'Content-Type': 'text/event-stream',
    });
    const decoder = new TextDecoder();
    const reader = stream.getReader();
    let closed = false;
    let responseText = '';
    while (!closed) {
      await reader.read().then(({ done, value }) => {
        if (done) {
          closed = true;
          res.end();
        } else {
          const text = decoder.decode(value);
          responseText += text;
          res.write(text);
        }
      });
    }
    const completionTokenCount = encoding.encode(responseText).length;
    await saveLlmUsage(userId, model.id, 'chat', {
      prompt: tokenCount,
      completion: completionTokenCount,
      total: tokenCount + completionTokenCount,
    });
  } catch (error) {
    console.error(error);
    const errorRes = getErrorResponseBody(error);
    res.status(500).json(errorRes);
  } finally {
    encoding.free();
  }
};

export default handler;
