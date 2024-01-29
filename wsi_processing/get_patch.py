from config import *
import multiprocessing
import util


def extract(main, slide_path, save_path):
    anno_path = slide_path[:-4] + 'xml'
    main._slide_setting(slide_path, anno_path, save_path)
    main.extract()

if __name__ == '__main__':
    # main setting
    main = util.ExtractPatch(PATCH_SIZE, RESOLUTION, DOWN_LEVEL, ANNO_RATIO, TISSUE_RATIO)

    # multi processing
    for i in range(0,len(SLIDE_LIST),BATCH):
        slide_batch = SLIDE_LIST[i:i+BATCH]
        for slide_path in slide_batch:
            p = multiprocessing.Process(target=extract, args=(main, slide_path, SAVE_PATH, ))
            p.start()
        p.join()
