package com.yupi.yuaiagent.rag;

import jakarta.annotation.Resource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;

import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * 恋爱大师向量数据库配置（懒加载模式，按需初始化）
 */
@Configuration
public class LoveAppVectorStoreConfig {

    private static final Logger log = LoggerFactory.getLogger(LoveAppVectorStoreConfig.class);
    
    // 用于标记是否已初始化
    private final AtomicBoolean initialized = new AtomicBoolean(false);
    
    // 缓存 VectorStore 实例
    private VectorStore cachedVectorStore;

    @Resource
    private LoveAppDocumentLoader loveAppDocumentLoader;

    @Resource
    private MyTokenTextSplitter myTokenTextSplitter;

    @Resource
    private MyKeywordEnricher myKeywordEnricher;

    /**
     * 创建懒加载的 VectorStore Bean
     * 只在第一次使用时才真正加载和增强文档
     */
    @Bean(name = "loveAppVectorStore")
    @Primary
    public VectorStore createLazyLoveAppVectorStore(EmbeddingModel dashscopeEmbeddingModel) {
        if (cachedVectorStore == null) {
            synchronized (this) {
                if (cachedVectorStore == null) {
                    log.info("首次初始化恋爱大师向量库...");
                    try {
                        // 创建基础的 VectorStore
                        SimpleVectorStore simpleVectorStore = SimpleVectorStore.builder(dashscopeEmbeddingModel).build();
                        
                        // 加载并处理文档
                        List<Document> documentList = loveAppDocumentLoader.loadMarkdowns();
                        log.info("已加载 {} 篇原始文档", documentList.size());
                        
                        // 自主切分文档
                        List<Document> splitDocuments = myTokenTextSplitter.splitCustomized(documentList);
                        log.info("文档切分为 {} 个片段", splitDocuments.size());
                        
                        // 自动补充关键词元信息（这里会调用 DashScope API）
                        log.info("开始为文档添加关键词元信息（可能需要几十秒）...");
                        List<Document> enrichedDocuments = myKeywordEnricher.enrichDocuments(splitDocuments);
                        log.info("关键词增强完成，共 {} 个文档", enrichedDocuments.size());
                        
                        // 添加到向量库
                        simpleVectorStore.add(enrichedDocuments);
                        cachedVectorStore = simpleVectorStore;
                        
                        log.info("恋爱大师向量库初始化完成！");
                    } catch (Exception e) {
                        log.error("初始化恋爱大师向量库失败：{}", e.getMessage(), e);
                        // 创建一个空的 VectorStore 避免后续调用失败
                        cachedVectorStore = SimpleVectorStore.builder(dashscopeEmbeddingModel).build();
                    }
                }
            }
        }
        return cachedVectorStore;
    }
}
