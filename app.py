# -*- coding: utf-8 -*-
from datetime import datetime
import uuid
from flask import Flask, request, jsonify, send_file, abort, current_app, send_from_directory
from flask_cors import CORS
import qdrant_client
import os
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient, models

# Flaskアプリケーションの初期化
app = Flask(__name__)
CORS(app)

collection_name = "unity_talk_test_data"
person_id="ID-002"
company_id="001"

openai.api_key = os.environ['OPENAI_API_KEY']
qdrant_api_key = os.environ['QDRANT_KEY']
qdrant_url_key = os.environ['QDRANT_URL']

points="- リハビリテーション分野におけるコミュニケーションスキルの重要性を強調し、英語力が国際的なコンテキストでどのように役立つか説明する。-演劇部で培った表現力やプレゼンテーションスキルが患者との相互作用やリハビリプログラムの企画にどのように活かせるかを示す。- 映画鑑賞から得られる創造力や脚本執筆での物語作りの経験をリハビリテーションのカスタマイズされたプログラム設計に応用する方法を紹介する。-ENFPの特性（社交的、直感的、感受性が強い、柔軟性がある）に合わせた多様でクリエイティブな学習環境や活動について話す。- **英語の応用**:- 言語療法や多言語患者とのコミュニケーションに英語がどう役立つかを強調する。- 英語を使った国際的な研究プロジェクトや学会でのプレゼンテーションの機会があることを述べる。- **演劇部の経験**:- 患者とのロールプレイやシミュレーション演習がカリキュラムに組み込まれていることを強調する。- 患者の物語を理解し、それに基づいたリハビリプログラムを設計する能力の重要性を説明する。- **創造性と物語作り**:- 症例研究やケースメソッドを用いた授業で、個々の患者に合わせたストーリーを構築する重要性について話す。-創造的なアプローチが求められるリハビリテーションのテクノロジーや療法の開発プロジェクトについて紹介する。- **ENFPの性格傾向**:-グループワークやディスカッションが充実している環境があることを示し、ENFPが活躍できる学びの場を提供する。-直観と感受性を生かして患者の感情を理解し、それに基づいたリハビリテーションプランを作成するプロセスについて説明する。"

student="[得意な科目]: 英語-[所属部活]: 演劇部-[特徴]: 映画鑑賞が趣味で、自分で脚本を書くことに興味を持っている"

feature="ENFP（社交的、直感的、感受性が強い、柔軟性がある）"

# OpenAIとLangChainの設定
embeddings = OpenAIEmbeddings()

client = qdrant_client.QdrantClient(
    qdrant_url_key,
    api_key=qdrant_api_key,
)

# Qdrantデータベースの設定
db = Qdrant(client=client, collection_name="univ_data", embeddings=embeddings)


@app.route('/submit-query', methods=['POST'])
def submit_query():
    data = request.json
    input_text = data.get('input_text', '')

    if not input_text:
        return jsonify({"error": "入力テキストが提供されていません"}), 400

    try:
        docs = db.similarity_search(query=input_text, k=4)
    
        contents = []
        for idx, i in enumerate(docs, 1):
          contents.append(f"情報{idx}：{i.page_content}")
        all_contents = " ーーー ".join(contents)
        response = openai.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "---あなたは、[専門家1]、[専門家2]、[専門家3]、の知識を持っています。[目的]を達成するために、[ルール１]、[ルール２]、[ルール３]、[ルール４]、[ルール５]、[ルール６]、[ルール７]、[ルール８]、[ルール９]、[ルール１０]に従って、[業務]を遂行してください。"},
            {"role": "user", "content": f"---#条件ーーー[専門家1]:和歌山リハビリテーション専門職大学の優秀な案内人、[専門家2]:プロのキャリアアドバイザー、[専門家3]:心理学の専門家/[目的]:[学生1]が和歌山リハビリテーション専門職大学に魅力を感じるように、[学生１]に合わせた提案や具体的なアドバイスをする。/[業務]：[学生１]からの[質問]に150字以内でできるだけ具体的に答える。/[ルール１]：[学生1]の特徴を考慮し、[観点]にある項目が伝わるように情報をカスタマイズする。/[ルール２]：回答は、以下の「情報1〜4」のみを参考にし、生成する。/[ルール３]：必要な情報が「情報１〜４」に不足している場合は、「申し訳ありません、現在私の学習している内容では、その質問にはお答えできません」と返答する。/[ルール４]:必要な情報が「情報１〜４」に不足しているかどうかの判断は、「情報１〜４」に明記されているかどうかで判断し、最も厳格に判断する。/[ルール５]：回答は「私」を使って、実際に語りかける形で行う。/[ルール６]：[学生1]を「あなた」と呼ぶ。/[ルール７]:回答は、以下の[フォーマット]を参考に生成し、言葉使いなども参考にする。[ルール８]:「情報１〜４」はあなたが学習している内容として使用し、回答に「情報１によると、」や、「情報１〜４には、」などという文言は含めない。[ルール９]:[観点]や[性格診断のタイプ]はあなたが学習している内容として使用し、回答に[性格診断のタイプ]のアルファベットは含めず、[性格診断のタイプ]は「あなた」と表現する。/[ルール１０]:回答は、[性格診断のタイプ]を最大限に考慮し、強調すべき点を工夫する。/[性格診断のタイプ]:{feature}/[フォーマット]:作業療法はあまり馴染みがないよね。でも、実は作業療法は私たちの生活を支えている重要な分野だよ。、、、/[観点]:{points}/[学生１]：{student}/[質問]："},
            {"role": "user", "content": input_text},
            {"role": "user", "content": "---"},
            {"role": "user", "content": all_contents},
        ]
    )
        response_text = response.choices[0].text.strip()

        # 応答をデータベースに保存（省略可能）

        return jsonify({"response": response_text})
    except Exception as e:
        return jsonify({"error": "応答の生成中にエラーが発生しました", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
