    # Cost Distribution Analysis Cross-Group
    ws[f'A{current_row}'] = "Cost Distribution Analysis:"
    ws[f'A{current_row}'].font = Font(bold=True)
    current_row += 1
    
    if len(analysis_data['costs']['comparison_other']) > 0:
        for r in dataframe_to_rows(analysis_data['costs']['comparison_other'], index=False, header=True):
            for c, value in enumerate(r, 1):
                ws.cell(row=current_row, column=c, value=value)
                if current_row == ws.max_row - len(analysis_data['costs']['comparison_other']):
                    ws.cell(row=current_row, column=c).font = Font(bold=True)
            current_row += 1

    # === ADD THESE LINES (insert charts) ===
    emb_path = create_embedding_comparison_chart(
        selected_row['outlier_pin'],
        selected_row['typical_same_pin'],
        selected_row['typical_other_pin'],
        selected_row['outlier_name'],
        selected_row['typical_same_name'],
        selected_row['typical_other_name']
    )
    ws.add_image(Image(emb_path), f"A{current_row}")
    current_row += 25  # adjust if needed

    pca_path = create_pca_chart(
        selected_row['outlier_pin'],
        selected_row['typical_same_pin'],
        selected_row['typical_other_pin'],
        selected_row['outlier_label'],
        selected_row['overlap_label'],
        selected_row['outlier_name'],
        selected_row['typical_same_name'],
        selected_row['typical_other_name']
    )
    ws.add_image(Image(pca_path), f"A{current_row}")
    current_row += 25  # adjust if needed
    # === END ADD ===
    
    # Memory cleanup
    del analysis_data
    gc.collect()
