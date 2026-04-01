"""
向量存储和检索模块
实现基于余弦相似度的向量搜索功能
"""

import json
import sqlite3
from typing import List, Tuple, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
import numpy as np
from memory_system.storage.models import Memory
from memory_system.utils.logger import get_storage_logger


class VectorStore:
    """
    向量存储类
    提供向量存储、检索和相似度计算功能
    """

    def __init__(self, db_path: str):
        """
        初始化向量存储

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.logger = get_storage_logger()

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        计算两个向量的余弦相似度

        Args:
            vec1: 向量1
            vec2: 向量2

        Returns:
            float: 相似度分数 (0-1)
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        # 计算点积
        dot_product = np.dot(vec1, vec2)

        # 计算模
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        # 避免除零
        if norm1 == 0 or norm2 == 0:
            return 0.0

        # 余弦相似度
        similarity = dot_product / (norm1 * norm2)

        # 确保结果在[0, 1]范围内
        similarity = float(max(0, min(1, similarity)))

        return similarity

    def search_similar_memories(
        self,
        query_embedding: List[float],
        session: Session,
        top_k: int = 5,
        threshold: float = 0.7,
        exclude_ids: Optional[List[int]] = None
    ) -> List[Tuple[Memory, float]]:
        """
        搜索相似的回忆

        Args:
            query_embedding: 查询向量
            session: 数据库会话
            top_k: 返回前K个结果
            threshold: 相似度阈值
            exclude_ids: 要排除的记忆ID列表

        Returns:
            List[Tuple[Memory, float]]: (记忆对象, 相似度分数) 列表
        """
        # 获取所有有向量的记忆
        query = session.query(Memory).filter(
            Memory.embedding.isnot(None),
            Memory.embedding != ""
        )

        # 排除指定的ID
        if exclude_ids:
            query = query.filter(~Memory.id.in_(exclude_ids))

        memories = query.all()

        # 计算相似度
        results = []
        for memory in memories:
            try:
                # 解析存储的向量
                memory_embedding = json.loads(memory.embedding)

                # 计算相似度
                similarity = self.cosine_similarity(query_embedding, memory_embedding)

                # 过滤低于阈值的
                if similarity >= threshold:
                    results.append((memory, similarity))
            except (json.JSONDecodeError, ValueError) as e:
                self.logger.warning(f"解析记忆 {memory.id} 的向量失败: {e}")
                continue

        # 按相似度排序并返回前K个
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def search_by_keywords(
        self,
        keywords: List[str],
        session: Session,
        top_k: int = 5
    ) -> List[Tuple[Memory, float]]:
        """
        基于关键词搜索记忆（使用向量相似度）

        Args:
            keywords: 关键词列表
            session: 数据库会话
            top_k: 返回前K个结果

        Returns:
            List[Tuple[Memory, float]]: (记忆对象, 相关性分数) 列表
        """
        # 这里需要先对关键词进行embedding，但由于需要LLM，所以这个方法
        # 将在IngestAgent或QueryAgent中实现具体逻辑
        # 这里只是提供一个框架

        # 暂时使用简单的文本匹配
        results = []
        memories = session.query(Memory).all()

        for memory in memories:
            score = 0.0
            content_lower = memory.content.lower()

            # 计算关键词匹配分数
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in content_lower:
                    score += 1.0

                # 在主题中匹配
                if memory.topics:
                    for topic in memory.topics:
                        if keyword_lower in topic.lower():
                            score += 0.5

                # 在实体中匹配
                if memory.entities:
                    for entity in memory.entities:
                        if keyword_lower in entity.lower():
                            score += 0.5

            # 归一化分数
            if score > 0:
                score = min(1.0, score / len(keywords))

            if score > 0:
                results.append((memory, score))

        # 按分数排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def calculate_similarity_matrix(
        self,
        embeddings: List[List[float]]
    ) -> List[List[float]]:
        """
        计算多个向量之间的相似度矩阵

        Args:
            embeddings: 向量列表

        Returns:
            List[List[float]]: 相似度矩阵
        """
        n = len(embeddings)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    matrix[i][j] = self.cosine_similarity(embeddings[i], embeddings[j])

        return matrix

    def find_clusters(
        self,
        session: Session,
        threshold: float = 0.8,
        min_cluster_size: int = 2
    ) -> List[List[int]]:
        """
        找出相似的记忆簇

        Args:
            session: 数据库会话
            threshold: 相似度阈值
            min_cluster_size: 最小簇大小

        Returns:
            List[List[int]]: 记忆ID簇的列表
        """
        # 获取所有有向量的记忆
        memories = session.query(Memory).filter(
            Memory.embedding.isnot(None),
            Memory.embedding != ""
        ).all()

        if not memories:
            return []

        # 解析向量
        embeddings = []
        for memory in memories:
            try:
                embedding = json.loads(memory.embedding)
                embeddings.append((memory.id, embedding))
            except json.JSONDecodeError:
                continue

        # 计算相似度矩阵
        n = len(embeddings)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                elif i < j:
                    similarity = self.cosine_similarity(
                        embeddings[i][1],
                        embeddings[j][1]
                    )
                    matrix[i][j] = similarity
                    matrix[j][i] = similarity

        # 简单的聚类算法（基于阈值）
        clusters = []
        visited = [False] * n

        for i in range(n):
            if visited[i]:
                continue

            cluster = [i]
            visited[i] = True

            # 查找所有与i相似的记忆
            for j in range(n):
                if not visited[j] and matrix[i][j] >= threshold:
                    cluster.append(j)
                    visited[j] = True

            # 如果簇大小满足要求，添加到结果
            if len(cluster) >= min_cluster_size:
                cluster_ids = [embeddings[idx][0] for idx in cluster]
                clusters.append(cluster_ids)

        return clusters

    @staticmethod
    def embedding_to_json(embedding: List[float]) -> str:
        """
        将向量转换为JSON字符串存储

        Args:
            embedding: 向量列表

        Returns:
            str: JSON字符串
        """
        return json.dumps(embedding)

    @staticmethod
    def embedding_from_json(json_str: str) -> List[float]:
        """
        从JSON字符串解析向量

        Args:
            json_str: JSON字符串

        Returns:
            List[float]: 向量列表
        """
        return json.loads(json_str)
