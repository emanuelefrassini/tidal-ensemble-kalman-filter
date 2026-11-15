from pathlib import Path
from tidal_enkf.data import read_series, load_observations, load_western_tide

ROOT=Path(__file__).resolve().parents[1]/"data"


def test_source_file_alignment():
    t,y=load_observations(ROOT,"waterlevel")
    tc,bc=load_western_tide(ROOT)
    assert len(t)==289 and y.shape==(289,4)
    assert (t==tc).all()
    assert len(bc)==289


def test_reader_metadata_ignored():
    t,v=read_series(ROOT/"tide_cadzand.txt")
    assert t[0].year==2013 and len(v)==289
