"""
This script contains utils function to perfrom Exploratory Data Analysis on the Extracted Features
"""

import pandas as pd
from typing import List, Union
from pathlib import Path
from FeatExtractorManager import FeatureRegistry
import re
import sys
from UmapExtractor import *



#============= USER_EDITABLE PART ============================================= 
sub_ids   = ['S01', 'S02', 'S03']                        # add as many users as you want
main_data_directory = Path(r"C:/Users/giusy/OneDrive/Desktop/PAPERS/2026_Sensors_speech/SilentWear/data")
wins_feats_name = 'wins_and_feats'
win_size_ms = [1400]                    # add as many as you want
sessions_to_consider = ['sess_1', 'sess_2']   
silent_analysis = True
vocalized_analysis = True

extract_umap = True
extract_pca = False
extract_tsne = False

consider_time_feats= True
consider_freq_feats = True
consider_wavelet_feats = True
#============= USER_EDITABLE PART ============================================= 


class Session_Feature_Analyzer:

    def __init__(self, 
                data_directory_wins_feats: Path, 
                subject_id: str, 
                win_size_ms: int, 
                session_ids: Union[str, List[str]],   # allow str or list
                silent_analysis : bool, 
                vocalized_analysis : bool,  
                
                extract_umap: bool,             
                extract_tsne : bool,            # not yet implemented   
                extract_pca : bool,             # not yet implemented

                consider_time_feats : bool, 
                consider_freq_feats : bool, 
                consider_wavelet_feats : bool, 

                ) -> None:

        
        """
        Performs Feature Analysis on a Per-Session Basis. 
        
        It plots per-batch feature space and per-session (aggregating session data)

        Arguments:
        data_directory_wins_feats: data directory where windows and features are stored (silent_wear/data/wins_and_feats)
        subject_id: id of the subject
        win_size_ms : int with window size to analyze
        session_id: str, session to consider
        silent_analysis: perform analysis on silent data
        vocalized_analysis: perform analysis on vocalized data

        extract_umap: True if we want to extract UMAP projections, else False
        extract_tnse : True if we want to extract TSNE projectstion, else False
        extract_pca : True if we want to perform PCA, else False

        consider_time_feats: True if we want to include Time Features, else False
        consider_freq_feats: True if we want to consider Frequency Features, else False
        consider_waveleft_feats : True if we want to consider Wavelet Transforms, else False

        
        """

        self.data_dire_wins_feats = data_directory_wins_feats
        self.sub_id = subject_id
        self.win_size_ms = win_size_ms
        self.session_ids = session_ids
        self.silent_analysis = silent_analysis
        self.vocalized_analysis = vocalized_analysis

        # To determine which features to use
        self.consider_time_feats = consider_time_feats
        self.consider_freq_feats = consider_freq_feats
        self.consider_wavelet_feats = consider_wavelet_feats
        self.features_to_consider = self.feature_names_to_consider()
        self.df_feat_cols_to_consider = None

        # To determine Type of Analysis
        self.extract_umap = extract_umap
        self.extract_pca = extract_pca
        self.extract_tsne = extract_tsne
    
    def find_feature_files(self):
        """
        Return (silent_files, vocalized_files) filtered by one or more session ids.
        If self.session_ids is empty/None, return all files.
        """
        base_silent = self.data_dire_wins_feats / self.sub_id / "silent" / f"WIN_{self.win_size_ms}"
        base_vocal  = self.data_dire_wins_feats / self.sub_id / "vocalized" / f"WIN_{self.win_size_ms}"

        h5_files_silent = sorted(base_silent.rglob("*.h5")) if base_silent.exists() else []
        h5_files_vocal  = sorted(base_vocal.rglob("*.h5"))  if base_vocal.exists()  else []

        # If user didn't specify sessions -> no filtering
        if not self.session_ids:
            return h5_files_silent, h5_files_vocal

        # Filter: keep files that contain ANY of the session strings
        def keep_file(p: Path) -> bool:
            s = str(p)
            return any(sess in s for sess in self.session_ids)

        h5_files_silent = [f for f in h5_files_silent if keep_file(f)]
        h5_files_vocal  = [f for f in h5_files_vocal  if keep_file(f)]

        return h5_files_silent, h5_files_vocal
    

        
    def feature_names_to_consider(self):
        """
        Returns the base feature names to consider depending on flags.
        """
        features = []

        if self.consider_time_feats:
            features += FeatureRegistry.TIME_DOMAIN

        if self.consider_freq_feats:
            features += FeatureRegistry.FREQUENCY_DOMAIN

        if self.consider_wavelet_feats:
            features += FeatureRegistry.WAVELET_DOMAIN

        return features
    
    def print_unique_features(self, df):
        """
        Prints all unique feature names present in the DataFrame columns,
        ignoring window and channel suffixes.
        """

        feature_names = set()

        for col in df.columns:
            # Match: <feature>_<win>_Ch_<id>_filt
            match = re.match(r"^(.+?)_\d+_Ch_\d+_filt$", col)

            if match:
                feature_names.add(match.group(1))

        feature_names = sorted(feature_names)

        print("\nUnique features found:")
        for f in feature_names:
            print(" -", f)

        print("\nTotal unique features:", len(feature_names))

        return feature_names

    def feature_columns_to_consider(self, df):
        """
        Returns only the DataFrame columns corresponding to selected feature groups.
        """

        # Step 1: Get base feature names
        features = self.feature_names_to_consider()
        print(features)

        if not features:
            raise ValueError("No feature groups selected!")

        # <feature>_<win>_Ch_<idx>_filt
        pattern = r"^(" + "|".join(map(re.escape, features)) + r")_\d+_Ch_\d+_filt$"


        # Step 3: Select matching columns
        selected_cols = [c for c in df.columns if re.search(pattern, c)]

        return selected_cols
        


    def main(self):
        """
        Docstring for main
        
        :param self: Description
        """

        # Find all files
        h5_files_silent, h5_files_vocalized = self.find_feature_files()
        # Collect dataframes
        dfs_silent = []
        dfs_vocal = []

        for files, condition in [(h5_files_silent, "silent"), (h5_files_vocalized, "vocalized")]:
            for file_path in files:
                df = pd.read_hdf(file_path, key="wins_feats")

                if self.df_feat_cols_to_consider is None:
                    self.df_feat_cols_to_consider = self.feature_columns_to_consider(df)

                df_feats = df[self.df_feat_cols_to_consider + ["Label_str", "batch_id", "session_id"]].copy()
                df_feats["condition"] = condition

                if condition == "silent":
                    dfs_silent.append(df_feats)
                else:
                    dfs_vocal.append(df_feats)

        df_silent_all = pd.concat(dfs_silent, ignore_index=True) if dfs_silent else None
        df_vocal_all  = pd.concat(dfs_vocal,  ignore_index=True) if dfs_vocal else None

        umap_extractor = UMAP_Projection_Extractor(
            config=UMAPConfig(n_neighbors=20, min_dist=0.05, max_points=5000),
            out_dir=self.data_dire_wins_feats.parent / Path("umap_plots") / self.sub_id / f"WIN_{self.win_size_ms}",
            subject_id=self.sub_id, 
        )
        
        # Feature columns are self.df_feat_cols_to_consider
        if df_silent_all is not None:
            # print("Processing Silent Data.")
            # print("Contains sessions:", df_silent_all['session_id'].unique(), "Batches:",df_silent_all['batch_id'].unique())
            # print("Total samples:", len(df_silent_all))
            umap_extractor.plot_per_session(df_silent_all, self.df_feat_cols_to_consider, condition="silent", show=False)
            umap_extractor.plot_across_sessions(df_silent_all, self.df_feat_cols_to_consider, condition="silent", show=False)

        if df_vocal_all is not None:
            umap_extractor.plot_per_session(df_vocal_all, self.df_feat_cols_to_consider, condition="vocalized", show=False)
            umap_extractor.plot_across_sessions(df_vocal_all, self.df_feat_cols_to_consider, condition="vocalized", show=False)




if __name__=='__main__':

    for sub_id in sub_ids:
        for win_size in win_size_ms:
            print("PROCESSING SUBJECT:", sub_id, "WIN SIZE", win_size, "Sessions:", sessions_to_consider)
            # Initialize Class
            session_feat_analyzer = Session_Feature_Analyzer(
                data_directory_wins_feats=main_data_directory/Path(wins_feats_name), 
                subject_id=sub_id, 
                session_ids=sessions_to_consider, 
                win_size_ms=win_size, 
                silent_analysis=silent_analysis, 
                vocalized_analysis=vocalized_analysis, 
                extract_umap=extract_umap, 
                extract_tsne=extract_tsne, 
                extract_pca=extract_pca, 
                consider_time_feats=consider_time_feats, 
                consider_freq_feats=consider_freq_feats, 
                consider_wavelet_feats=consider_wavelet_feats)
            session_feat_analyzer.main()

            



            
    


    
    
