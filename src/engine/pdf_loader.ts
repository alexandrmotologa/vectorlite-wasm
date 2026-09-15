import * as pdfjsLib from 'pdfjs-dist';

// Configure worker URL for pdfjs-dist
if (typeof window !== 'undefined' && 'Worker' in window) {
  pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.mjs`;
}

export interface PDFExtractResult {
  text: string;
  pageCount: number;
  metadata: {
    title?: string;
    author?: string;
    creator?: string;
  };
}

/**
 * Extracts plain text from a PDF file client-side.
 */
export async function extractTextFromPDF(file: File | ArrayBuffer): Promise<PDFExtractResult> {
  const data = file instanceof File ? await file.arrayBuffer() : file;
  const loadingTask = pdfjsLib.getDocument({ data });
  const pdf = await loadingTask.promise;

  let fullText = '';
  const numPages = pdf.numPages;

  for (let pageNum = 1; pageNum <= numPages; pageNum++) {
    const page = await pdf.getPage(pageNum);
    const textContent = await page.getTextContent();
    const pageText = textContent.items
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      .map((item: any) => item.str)
      .join(' ');

    fullText += `\n\n--- Page ${pageNum} ---\n\n` + pageText;
  }

  let metadata = {};
  try {
    const meta = await pdf.getMetadata();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const info = meta?.info as any;
    metadata = {
      title: info?.Title,
      author: info?.Author,
      creator: info?.Creator,
    };
  } catch {
    // Metadata reading failure is non-fatal
  }

  return {
    text: fullText.trim(),
    pageCount: numPages,
    metadata,
  };
}
