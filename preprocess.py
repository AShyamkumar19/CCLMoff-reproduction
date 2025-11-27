import os
import subprocess
import pandas as pd

def run_cas_offinder(sgRNA_sequence, genome_file, raw_output_file, mismatches=3, device="G0", pam="NGG"):
    """
    Run Cas-OFFinder to generate all possible sgRNA-target site mismatch pairs.
    """
    guide_with_pam = f"{sgRNA_sequence}{pam}"

    # Build Cas-OFFinder input template
    with open("cas_offinder_input.txt", "w") as f:
        f.write(f"{genome_file}\n")
        f.write(f"{'N' * len(guide_with_pam)} 0 0\n")
        f.write(f"{guide_with_pam} {mismatches}\n")
    
    command = [
        "cas-offinder",
        "cas_offinder_input.txt",
        device,
        raw_output_file
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"[INFO] Cas-OFFinder completed. Raw output saved to {raw_output_file}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error running Cas-OFFinder: {e}")
    finally:
        if os.path.exists("cas_offinder_input.txt"):
            os.remove("cas_offinder_input.txt")


def convert_cas_to_cclmoff(raw_output_file, output_tsv):
    """
    Convert Cas-OFFinder output into CCLMoff input format:
    Columns: sgRNA_seq, off_seq, read, pred, label
    """
    print(f"[INFO] Converting Cas-OFFinder output → {output_tsv}")

    rows = []
    with open(raw_output_file, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue

            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue

            # Cas-OFFinder columns:
            # Id, BulgeType, crRNA, DNA, Chrom, Loc, Dir, Mismatches, BulgeSize
            _, _, sgRNA, off_seq, *_ = parts

            rows.append({
                "sgRNA_seq": sgRNA,
                "off_seq": off_seq,
                "read": 0,
                "pred": 0,
                "label": 0
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_tsv, sep="\t", index=False)

    print(f"[SUCCESS] CCLMoff-format file created: {output_tsv}")
    print(f"[INFO] Total pairs: {len(df)}")


def preprocess_sgRNA(sgRNA_sequence, genome_file, output_file, mismatches=3, device="G0", pam="NGG"):
    """
    Run Cas-OFFinder AND convert results into CCLMoff-ready TSV.
    """
    raw_output = "cas_raw_output.txt"
    print(f"[INFO] Preprocessing sgRNA: {sgRNA_sequence}")

    # Step 1 — Run Cas-OFFinder
    run_cas_offinder(sgRNA_sequence, genome_file, raw_output, mismatches, device, pam)

    # Step 2 — Convert to CCLMoff format
    convert_cas_to_cclmoff(raw_output, output_file)

    print("[DONE] Preprocessing complete.")


if __name__ == "__main__":
    sgRNA_sequence = "GAGTCCGAGCAGAAGAAGAA"
    genome_file = "/mnt/c/Users/aarav/Documents/College Shi/Fall 2025/Bioinformatics/CCLMoff/genomes/hg38.fa"
    output_file = "/mnt/c/Users/aarav/Documents/College Shi/Fall 2025/Bioinformatics/CCLMoff/data/test_cclmoff.tsv"
    mismatches = 3  

    preprocess_sgRNA(sgRNA_sequence, genome_file, output_file, mismatches)
