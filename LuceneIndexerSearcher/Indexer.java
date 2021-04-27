package sf.index;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Paths;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field.Store;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexWriterConfig.OpenMode;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;


public class Indexer {
	
	/**
	 * Build index for all documents in a file. 
	 * Each document per line is in the form of "doc's id ===> doc's content.
	 */
	public static void buildIndex4DocsInFile(String docsFile, String indexPath) {
		
		File f = new File(indexPath);
		if (!f.exists())
			f.mkdirs();

		try {

			Directory dir = FSDirectory.open(Paths.get(indexPath));
			Analyzer analyzer = new WhitespaceAnalyzer();
			IndexWriterConfig iwc = new IndexWriterConfig(analyzer);
			iwc.setOpenMode(OpenMode.CREATE);
			IndexWriter writer = new IndexWriter(dir, iwc);

			String line = "";
			int c = 0;
			BufferedReader br = new BufferedReader(new FileReader(docsFile));
			while ((line = br.readLine()) != null) {
				c++;
				System.out.println(c + " ***** " + line);
				String[] sa = line.split("===>");
				Document doc = new Document();
				doc.add(new StringField("id", sa[0].trim(), Store.YES));
				doc.add(new TextField("contents", sa[1].trim(), Store.NO));
				writer.addDocument(doc);
			}
			br.close();
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		
		String projectPath = "D:\\Research\\Experiments\\ShellFusion";
		String modelsPath = projectPath + "\\models";
		buildIndex4DocsInFile(modelsPath + "\\lucene_docs.txt", modelsPath + "\\lucene_index");
	}
	
}
