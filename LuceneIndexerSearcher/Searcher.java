package sf.index;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.nio.file.Paths;

import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.FSDirectory;


public class Searcher {
	
	
	/**
	 * Search the top-k similar documents for a query.
	 * @param indexPath: the path that contains the LUCENE index built for documents.
	 * @param query: a query processed using the same algorithm adopted to process documents before indexing.
	 * @param k: the top k value.
	 * @param resFile: the file to store the top k similar documents.
	 */
	public static void searchTopk4Query(String pQuery, 
			String indexPath, int k, String resFile) {

		try {

			IndexReader reader = DirectoryReader.open(FSDirectory.open(Paths.get(indexPath)));
			IndexSearcher searcher = new IndexSearcher(reader);
			QueryParser parser = new QueryParser("contents", new WhitespaceAnalyzer());
			
			long start = System.currentTimeMillis();
			Query query = parser.parse(pQuery);
			TopDocs results = searcher.search(query, k);
			ScoreDoc[] hits = results.scoreDocs;
			System.out.println(System.currentTimeMillis() - start);
			
			String topkSimDocs = "";
			for (ScoreDoc hit : hits) {
				Document doc = searcher.doc(hit.doc);
				topkSimDocs += doc.get("id") + "\t" + hit.score + "\n";
			}
			BufferedWriter bw = new BufferedWriter(new FileWriter(resFile));
			bw.write(topkSimDocs.trim());
			bw.close();
			reader.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	
	/**
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		
		String projectPath = "D:\\Research\\Experiments\\ShellFusion";
		String indexPath = projectPath + "\\models\\lucene_index";
		String resFile = projectPath + "\\_test\\lucene_topN.txt";
		String pQuery = "creat singl pdf multipl text imag pdf file";  // a preprocessed query
		
		searchTopk4Query(pQuery, indexPath, 1000, resFile);
	}
	
}
