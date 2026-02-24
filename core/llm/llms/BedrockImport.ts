import {
  BedrockRuntimeClient,
  InvokeModelWithResponseStreamCommand,
} from "@aws-sdk/client-bedrock-runtime";
import { fromNodeProviderChain } from "@aws-sdk/credential-providers";
import { NodeHttpHandler } from "@smithy/node-http-handler";
import { Agent as HttpAgent } from "http";
import { HttpProxyAgent } from "http-proxy-agent";
import { Agent as HttpsAgent } from "https";
import { HttpsProxyAgent } from "https-proxy-agent";

import { CompletionOptions, LLMOptions } from "../../index.js";
import { BaseLLM } from "../index.js";

class BedrockImport extends BaseLLM {
  static providerName = "bedrockimport";
  static defaultOptions: Partial<LLMOptions> = {
    region: "us-east-1",
  };
  // the BedRock imported custom model ARN
  modelArn?: string | undefined;

  constructor(options: LLMOptions) {
    super(options);
    if (!options.apiBase) {
      this.apiBase = `https://bedrock-runtime.${options.region}.amazonaws.com`;
    }
    if (options.modelArn) {
      this.modelArn = options.modelArn;
    }
    if (options.profile) {
      this.profile = options.profile;
    } else {
      this.profile = "bedrock";
    }
  }

  protected async *_streamComplete(
    prompt: string,
    signal: AbortSignal,
    options: CompletionOptions,
  ): AsyncGenerator<string> {
    const credentials = await this._getCredentials();
    const client = new BedrockRuntimeClient({
      region: this.region,
      endpoint: this.apiBase,
      requestHandler: this._createRequestHandler(),
      credentials: {
        accessKeyId: credentials.accessKeyId,
        secretAccessKey: credentials.secretAccessKey,
        sessionToken: credentials.sessionToken || "",
      },
    });

    const input = this._generateInvokeModelCommandInput(prompt, options);
    const command = new InvokeModelWithResponseStreamCommand(input);
    const response = await client.send(command, { abortSignal: signal });

    if (response.body) {
      for await (const item of response.body) {
        const decoder = new TextDecoder();
        const decoded = decoder.decode(item.chunk?.bytes);
        try {
          const chunk = JSON.parse(decoded);
          if (chunk.outputs[0].text) {
            yield chunk.outputs[0].text;
          }
        } catch (e) {
          throw new Error(`Malformed JSON received from Bedrock: ${decoded}`);
        }
      }
    }
  }

  private _generateInvokeModelCommandInput(
    prompt: string,
    options: CompletionOptions,
  ): any {
    const payload = {
      prompt: prompt,
    };

    return {
      body: JSON.stringify(payload),
      modelId: this.modelArn,
      accept: "application/json",
      contentType: "application/json",
    };
  }

  private async _getCredentials() {
    try {
      return await fromNodeProviderChain({
        profile: this.profile,
        ignoreCache: true,
      })();
    } catch (e) {
      console.warn(
        `AWS profile with name ${this.profile} not found in ~/.aws/credentials, using default profile`,
      );
      return await fromNodeProviderChain()();
    }
  }

  private _getProxyForProtocol(protocol: string): string | undefined {
    if (this.requestOptions?.proxy) {
      return this.requestOptions.proxy;
    }
    if (protocol === "https:") {
      return (
        process.env.HTTPS_PROXY ||
        process.env.https_proxy ||
        process.env.HTTP_PROXY ||
        process.env.http_proxy
      );
    }
    return process.env.HTTP_PROXY || process.env.http_proxy;
  }

  private _shouldBypassProxy(hostname: string): boolean {
    const noProxy = [
      ...(process.env.NO_PROXY || process.env.no_proxy || "")
        .split(",")
        .map((item: string) => item.trim().toLowerCase())
        .filter((item: string) => !!item),
      ...(this.requestOptions?.noProxy ?? [])
        .map((item: string) => item.trim().toLowerCase())
        .filter((item: string) => !!item),
    ];

    const normalizedHostname = hostname.toLowerCase();
    return noProxy.some((pattern) => {
      const [hostWithoutPort, hostPort] = normalizedHostname.split(":");
      const [patternWithoutPort, patternPort] = pattern.split(":");

      if (patternPort && (!hostPort || hostPort !== patternPort)) {
        return false;
      }

      if (patternWithoutPort === hostWithoutPort) {
        return true;
      }

      if (
        patternWithoutPort.startsWith("*.") &&
        hostWithoutPort.endsWith(patternWithoutPort.substring(1))
      ) {
        return true;
      }

      if (
        patternWithoutPort.startsWith(".") &&
        hostWithoutPort.endsWith(patternWithoutPort.slice(1))
      ) {
        return true;
      }

      return false;
    });
  }

  private _createRequestHandler() {
    const endpointUrl = new URL(this.apiBase!);
    const proxy = this._getProxyForProtocol(endpointUrl.protocol);

    if (!proxy || this._shouldBypassProxy(endpointUrl.hostname)) {
      return undefined;
    }

    const requestTimeoutMs = this.requestOptions?.timeout
      ? this.requestOptions.timeout * 1000
      : undefined;

    const httpAgent: HttpAgent | HttpsAgent =
      endpointUrl.protocol === "https:"
        ? new HttpsProxyAgent(proxy)
        : new HttpProxyAgent(proxy);

    return new NodeHttpHandler({
      httpAgent,
      httpsAgent: httpAgent,
      requestTimeout: requestTimeoutMs,
    });
  }
}

export default BedrockImport;
