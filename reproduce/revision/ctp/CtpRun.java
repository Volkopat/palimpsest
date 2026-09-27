/*
 * Headless driver for the RSNA CTP anonymizer, written for the reviewer-response revision.
 *
 * Both reviewers asked for a fairly configured CTP arm: the published baseline ran CTP's
 * metadata anonymizer only, with no pixel script, which makes its 0/35 burned-in recall a
 * statement about configuration rather than about CTP. This driver runs the same metadata
 * script with and without CTP's own DicomPixelAnonymizer, so the difference between the two
 * arms is attributable to the pixel component alone.
 *
 * CTP ships as a GUI installer; the classes underneath are ordinary library calls, so this
 * calls them directly rather than driving a GUI.
 *
 *   DICOMAnonymizer.anonymize(in, out, cmds, lkup, integerTable, forceIVRLE, renameToSOPIUID)
 *   PixelScript.getMatchingSignature(DicomObject) -> Signature.regions
 *   DICOMPixelAnonymizer.anonymize(in, out, regions, decompress, setBurnedInAnnotation)
 *
 * Usage:
 *   java -cp <CTP libs>:. CtpRun <inputRoot> <outputRoot> <daScript> <pixelScript|NONE> <workDir> [limit]
 */
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import org.rsna.ctp.objects.DicomObject;
import org.rsna.ctp.stdstages.anonymizer.AnonymizerStatus;
import org.rsna.ctp.stdstages.anonymizer.IntegerTable;
import org.rsna.ctp.stdstages.anonymizer.dicom.DAScript;
import org.rsna.ctp.stdstages.anonymizer.dicom.DICOMAnonymizer;
import org.rsna.ctp.stdstages.anonymizer.dicom.DICOMPixelAnonymizer;
import org.rsna.ctp.stdstages.anonymizer.dicom.PixelScript;
import org.rsna.ctp.stdstages.anonymizer.dicom.Signature;

public class CtpRun {

    static void collect(File dir, List<File> out) {
        File[] kids = dir.listFiles();
        if (kids == null) return;
        for (File f : kids) {
            if (f.isDirectory()) collect(f, out);
            else if (f.getName().toLowerCase().endsWith(".dcm")) out.add(f);
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 5) {
            System.err.println("usage: CtpRun <inputRoot> <outputRoot> <daScript> <pixelScript|NONE> <workDir> [limit]");
            System.exit(2);
        }
        File inRoot = new File(args[0]);
        File outRoot = new File(args[1]);
        File daFile = new File(args[2]);
        String pixArg = args[3];
        File workDir = new File(args[4]);
        int limit = args.length > 5 ? Integer.parseInt(args[5]) : 0;

        workDir.mkdirs();
        Properties cmds = DAScript.getInstance(daFile).toProperties();
        Properties lkup = new Properties();
        IntegerTable itab = new IntegerTable(workDir);
        PixelScript pixScript = null;
        if (!pixArg.equalsIgnoreCase("NONE")) {
            pixScript = new PixelScript(new File(pixArg));
            if (pixScript.toString() == null) System.out.println("[warn] pixel script loaded but empty");
        }
        System.out.println("[init] script=" + daFile.getName()
                + " properties=" + cmds.size()
                + " pixelScript=" + (pixScript == null ? "NONE" : new File(pixArg).getName()));

        List<File> files = new ArrayList<File>();
        collect(inRoot, files);
        java.util.Collections.sort(files);
        if (limit > 0 && limit < files.size()) files = files.subList(0, limit);
        System.out.println("[init] " + files.size() + " input instances");

        int ok = 0, fail = 0, pixApplied = 0, pixNoSig = 0, pixFail = 0;
        long t0 = System.currentTimeMillis();
        int inLen = inRoot.getAbsolutePath().length();

        for (int i = 0; i < files.size(); i++) {
            File in = files.get(i);
            String rel = in.getAbsolutePath().substring(inLen);
            File out = new File(outRoot, rel);
            out.getParentFile().mkdirs();

            // Order matters and is easy to get wrong. The pixel signatures are matched on
            // identifying tags (Manufacturer, SeriesDescription, ImageType), and the metadata
            // profile blanks exactly those, so testing for a signature after anonymizing would
            // match nothing and silently report that CTP's pixel anonymizer never fires. CTP
            // pipelines therefore place the DicomPixelAnonymizer stage BEFORE the
            // DicomAnonymizer stage, and so do we: pixels first, from the original object.
            File metaIn = in;
            File pixTmp = null;
            if (pixScript != null) {
                try {
                    DicomObject dob = new DicomObject(in);
                    Signature sig = pixScript.getMatchingSignature(dob);
                    dob.close();
                    if (sig != null && sig.regions != null && sig.regions.size() > 0) {
                        pixTmp = new File(out.getAbsolutePath() + ".pix");
                        AnonymizerStatus ps = DICOMPixelAnonymizer.anonymize(in, pixTmp, sig.regions, false, true);
                        if (ps.isOK() && pixTmp.exists()) {
                            metaIn = pixTmp; pixApplied++;
                            // Record what actually matched, so "the pixel stage fires on N of
                            // M" can be reported as a fact about which images rather than a
                            // bare count.
                            DicomObject d2 = new DicomObject(in);
                            System.out.println("[pix] " + d2.getElementValue("Modality")
                                    + " | " + d2.getElementValue("Manufacturer")
                                    + " | " + d2.getElementValue("SeriesDescription"));
                            d2.close();
                        }
                        else { if (pixTmp.exists()) pixTmp.delete(); pixTmp = null; pixFail++; }
                    } else pixNoSig++;
                } catch (Throwable t) { pixFail++; }
            }

            try {
                AnonymizerStatus st = DICOMAnonymizer.anonymize(metaIn, out, cmds, lkup, itab, false, false);
                if (st.isOK()) ok++; else fail++;
            } catch (Throwable t) { fail++; }
            if (pixTmp != null && pixTmp.exists()) pixTmp.delete();

            if ((i + 1) % 2000 == 0)
                System.out.println("  " + (i + 1) + "/" + files.size()
                        + "  ok=" + ok + " fail=" + fail
                        + " pixApplied=" + pixApplied
                        + "  " + ((System.currentTimeMillis() - t0) / 1000) + "s");
        }
        itab.close();
        System.out.println("[done] instances=" + files.size() + " ok=" + ok + " fail=" + fail
                + " pixApplied=" + pixApplied + " pixNoSignature=" + pixNoSig + " pixFail=" + pixFail
                + "  " + ((System.currentTimeMillis() - t0) / 1000) + "s");
    }
}
