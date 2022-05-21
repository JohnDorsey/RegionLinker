
import os
import shutil
import itertools

# from PureGenTools import higher_range_by_corners


#os.symlink
#  won't overwrite anything by default.
#os.readlink
#  follows ALL the links in separate element levels, but not in the same element level. so os.readlink(os.readlink(A)) is not always equal to os.readlink(A)
#os.read
#os.statvfs
#  doesn't see symlinks that point to nothing, but readlink does.
"""
bash:
  rm file -> deletes file
  rm symlink -> deletes linked file
  unlink file -> deletes file (actually, removes the pointer to it, so programs that are working with the file continue to work with it, but after they let it go, it really is gone).
  unlink symlink -> deletes link in the same way it deletes files. doesn't affect the linked file.
"""

def assert_equal(thing0, thing1):
    assert thing0 == thing1, "{} does not equal {}.".format(thing0, thing1)


    
def gen_directly_adjacent_int_vecs(input_int_vec):
    for i in range(len(input_int_vec)):
        for diffSign in (-1, 1):
            yield input_int_vec[0:i] + (input_int_vec[i]+diffSign,) + input_int_vec[i+1:]

assert set(gen_directly_adjacent_int_vecs((5,-7))) == {(4,-7), (6,-7), (5,-6), (5,-8)}
            

def gen_words_in_alphabet(alphabet, length):
    return itertools.product(*([list(alphabet)]*length))

"""
def gen_selectors_from_compressed(uncompressed, compressed):
    compressedGen = iter(compressed)
    currentCompressedItem = next(compressedGen)
    for item in uncompressed:
        if item == currentCompressedItem:
            yield 1
            currentCompressedItem = next(compressedGen)
        else:
            yield 0
    assert False, "probably couldn't find all items, last was {}.".format(currentCompressedItem)

def gen_track_delayed_true_count(data):
    count = 0
    for item in data:
        yield count
        if item:
            count += 1

def selectors_from_indices(indices):
    assert isinstance(indices, set)
    for i in itertools.count():
        if i in indices:
            yield 1
        else:
            yield 0
"""
# def gen_place_at_indices


def gen_all_hybrids(*basis_outfits):
    # a basis outfit is like [formal shirt, formal pants, formal shoes]. all provided basis outfits are the same length. this generator gives all possible outfits by choosing what goes in each position from the options given by [items in that same position in each of the basis outfits].
    return itertools.product(*zip(*basis_outfits))
    
assert_equal(set(gen_all_hybrids([0],[1])), {(0,), (1,)})
    
    
def gen_hypercube_corners(corner0, corner1):
    assert len(corner0) == len(corner1)
    for i in range(len(corner0)):
        assert corner0[i] != corner1[i]
    return gen_all_hybrids(corner0, corner1)
    
assert_equal(set(gen_hypercube_corners((4,9),(6,11))), {(4,9),(4,11),(6,9),(6,11)})



def gen_permutations_with_fill(data, fill_item, length):
    assert length >= len(data)
    return itertools.permutations(list(data)+([fill_item]*(length-len(data))))


def gen_partially_adjacent_int_vecs(input_int_vec, min_tweaks=1, max_tweaks=None):
    if max_tweaks is None:
        max_tweaks = len(input_int_vec)
    assert min_tweaks <= max_tweaks, "bad range description."
    for tweakCount in range(min_tweaks, max_tweaks+1):
        """
        for tweakIndices in itertools.combinations(range(len(input_int_vec)), tweakCount):
            for tweakSigns in gen_words_in_alphabet([-1,1], tweakCount):
                # yield tuple(inputIntVecItem + selector*tweakSigns[tweakIndex] for inputIntVecItem, (tweakIndex, selector) in zip(input_int_vec, gen_track_delayed_true_count(gen_selectors_from_compressed(range(len(input_int_vec)), tweakIndices))))
        """
        for tweakSigns in gen_words_in_alphabet([-1,1], tweakCount):
            for componentDifferences in gen_permutations_with_fill(tweakSigns, 0, len(input_int_vec)):
                yield tuple(component+componentDifference for component, componentDifference in zip(input_int_vec, componentDifferences))

assert_equal(set(gen_partially_adjacent_int_vecs((5,10,15), max_tweaks=2)), set(itertools.product(range(4,7),range(9,12),range(14,17))).difference(set(gen_hypercube_corners((4,9,14),(6,11,16))), {(5,10,15)}))
assert_equal(set(gen_partially_adjacent_int_vecs((5,10,15,20), max_tweaks=3)), set(itertools.product(range(4,7),range(9,12),range(14,17),range(19,22))).difference(set(gen_hypercube_corners((4,9,14,19),(6,11,16,21))), {(5,10,15,20)}))


def gen_allocation_permutations(bucket_count, resource_count, min_allocation=0, max_allocation=None, combination_mode=False):
    # like o o|o o o|o|o and such.
    if max_allocation is None:
        max_allocation = resource_count
    assert min_allocation >= 0
    assert bucket_count > 0
    assert resource_count >= bucket_count * min_allocation
    assert resource_count <= bucket_count * max_allocation, (bucket_count, resource_count, min_allocation, max_allocation)
    assert max_allocation >= 0
    assert max_allocation >= min_allocation
    if bucket_count == 1:
        yield (resource_count,)
        return
    else:
        reservedForRemainder = (bucket_count-1)*min_allocation
        for firstBucketSize in range(min_allocation, min(max_allocation, resource_count-reservedForRemainder)+1):
            remainderMaxAllocation = firstBucketSize if combination_mode else max_allocation
            remainderTotalCapacity = (bucket_count-1)*remainderMaxAllocation
            if remainderTotalCapacity + firstBucketSize < resource_count:
                continue
            for remainderTuple in gen_allocation_permutations(bucket_count-1, resource_count-firstBucketSize, min_allocation=min_allocation, max_allocation=remainderMaxAllocation, combination_mode=combination_mode):
                result = (firstBucketSize,)+remainderTuple
                yield result
                # print("{}: {}".format((bucket_count, resource_count, min_allocation, max_allocation), result))

assert_equal(list(gen_allocation_permutations(2,3)), [(0,3), (1,2), (2,1), (3,0)])
assert_equal(list(gen_allocation_permutations(3,3)), [(0,0,3), (0,1,2), (0,2,1), (0,3,0), (1,0,2), (1,1,1), (1,2,0), (2,0,1), (2,1,0), (3,0,0)])

assert_equal(list(gen_allocation_permutations(3,3,combination_mode=True)), [(1,1,1), (2,1,0), (3,0,0)])
assert_equal(list(gen_allocation_permutations(3,4,combination_mode=True)), [(2,1,1), (2,2,0), (3,1,0), (4,0,0)])

    

def interleave(*input_seqs):
    inputGens = [iter(seq) for seq in input_seqs]
    while len(inputGens) > 0:
        i = 0
        while i < len(inputGens):
            try:
                nextItem = next(inputGens[i])
                yield nextItem
                i += 1
            except StopIteration:
                inputGens.pop(i)
                

def gen_unique_list_spacings(data, length, fill_item=None):
    assert length >= len(data)
    for spaceSizes in gen_allocation_permutations(len(data)+1, length-len(data)):
        assert len(spaceSizes) == len(data) + 1
        yield list(itertools.chain.from_iterable(interleave((itertools.repeat(fill_item, spaceSize) for spaceSize in spaceSizes), ([item] for item in data))))


def gen_manhattan_bled_int_vecs_old(input_int_vec, *, min_distance=1, max_distance=None):
    assert min_distance >= 1, "not supported yet"
    assert max_distance >= min_distance
    for manhattanDistance in range(min_distance, max_distance+1):
        for tweakCount in range(1, min(len(input_int_vec)+1, manhattanDistance+1)): # since tweak size is 1, there can't be more tweaks than units of manhattan distance.
            for tweakMagnitudes in gen_allocation_permutations(tweakCount, manhattanDistance, min_allocation=1):
                for tweakSigns in gen_words_in_alphabet([-1,1], tweakCount):
                    tweakValues = [tweakMagnitude*tweakSign for tweakMagnitude, tweakSign in zip(tweakMagnitudes, tweakSigns)]
                    for componentDifferences in gen_unique_list_spacings(tweakValues, len(input_int_vec), fill_item=0):
                        yield tuple(component+componentDifference for component, componentDifference in zip(input_int_vec, componentDifferences))



def gen_manhattan_bled_int_vecs(input_int_vec, *, min_distance=1, max_distance=None):
    assert min_distance >= 1, "not supported yet"
    assert max_distance >= min_distance
    for manhattanDistance in range(min_distance, max_distance+1):
        for axialRangeRadii in gen_allocation_permutations(len(input_int_vec), manhattanDistance):
            # range(0-axialRangeSize//2, 1+axialRangeSize//2)
            #for offsetVec in itertools.product(*[ for axialRangeSize in axialRangeSizes]):
            #    yield tuple(component + componentDifference for component, componentDifference in
            for outputVec in itertools.product(*[([component-axialRangeRadius, component+axialRangeRadius] if axialRangeRadius > 0 else [component]) for component, axialRangeRadius in zip(input_int_vec, axialRangeRadii)]):
                yield outputVec

assert_equal(set(gen_manhattan_bled_int_vecs((5,10,15), max_distance=1)), {(5,10,14), (5,10,16), (5,9,15), (5,11,15), (4,10,15), (6,10,15)})
assert_equal(len(set(gen_manhattan_bled_int_vecs((5,10,15), max_distance=2))), 6+(6+12))

assert_equal(set(gen_manhattan_bled_int_vecs_old((5,10,15,20), max_distance=5)), set(gen_manhattan_bled_int_vecs((5,10,15,20), max_distance=5)))


def higher_range_by_corners(start_corner, stop_corner_inclusive, step_corner=None):
    if step_corner is None:
        step_corner = [(1 if start <= stop else -1) for start, stop in zip(start_corner, stop_corner_inclusive)]
    return itertools.product(*[range(start, stop+step, step) for start, stop, step in zip(start_corner, stop_corner_inclusive, step_corner)])


def gen_box_bled_int_vecs(input_int_vec, *, min_distance=0, max_distance=None):
    assert min_distance == 0
    assert max_distance >= min_distance
    # assert max_distance > 0
    return higher_range_by_corners([component-max_distance for component in input_int_vec], [component+max_distance for component in input_int_vec])
    
assert_equal(len(set(gen_box_bled_int_vecs((5,10,15,20), max_distance=3))), 7**4)











def get_set_expansion_using_multi_mutation(input_set, expansion_fun, exclusion_set=None):
    if exclusion_set is None:
        exclusion_set = set()
    expansionItemGroupGen = (expansion_fun(oldItem) for oldItem in input_set)
    expansionItemGen = itertools.chain.from_iterable(expansionItemGroupGen)
    expansionItemSet = set(expansionItemGen)
    culledExpansionItemSet = set(item for item in expansionItemSet if (item not in input_set and item not in exclusion_set))
    return culledExpansionItemSet
    
assert set(get_set_expansion_using_multi_mutation({(0,-7), (5,-7), (4,-7), (50,-70)}, gen_directly_adjacent_int_vecs, exclusion_set={(50,-71)})) == {
        (-1,-7), (1,-7), (0,-6), (0,-8),
                 (6,-7), (5,-6), (5,-8),
        (3,-7),         (4,-6), (4,-8),
        (49,-70), (51,-70), (50,-69),
    }

    
def get_set_expansion_stages_using_multi_mutation(input_set, expansion_fun, stage_count):
    #this method doesn't keep a growing exclusion set of all previous stages. It assumes that A -> B -> C -> A can't ever happen. in the case of incrementing and decremnting an integer coord, that's true. I think. Because the set of functions going into the expansion_fun contains the inverse of every function in it, there can be no one-way loops.
    assert stage_count > 0
    # if exclusion_set is None:
    #    exclusion_set= set()
    stages = []
    for stageIndex in range(0, stage_count+1):
        if stageIndex == 0:
            stages.append(input_set)
        else:
            stages.append(get_set_expansion_using_multi_mutation(stages[-1], expansion_fun, exclusion_set=stages[max(len(stages)-2,0)]))
    assert len(stages) == stage_count+1
    assert stages[0] == input_set
    return stages[1:]
"""                   4
   3                 434
  323               43234
 32123             4321234
  323               43234
   3   n=3.  a=13    434
                      4  n=4.  a(n)=2*n+2*(n-2)+a(n-1). a(4)=8+4+13=25. a(n)?=n*n+(n-1)*(n-1)
                      /* ...=n*n+n*n-n-n+1=2nn-2n+1=(n-1)(2n)+1 */
                      (n**2+(n-1)**2=(n-1)**2+(n-2)**2+4n-2) -> (n**2=(n-2)**2+4n-2)
                                                                 n**2=n**2-2nk-k**2
"""
assert [len(item) for item in get_set_expansion_stages_using_multi_mutation({(5,10)}, gen_directly_adjacent_int_vecs, 4)] == [4,8,12,16]







    

def chain_string_partitions(string, delimiters, include_delimiters=True, tolerate_failures=False, include_empty=True):
    result = [string]
    for delimiter in delimiters:
        assert delimiter != ''
        currentStepData = result[-1].partition(delimiter)
        if not tolerate_failures:
            if currentStepData[1] == '':
                assert False, "failed to find delimiter: {}.".format(delimiter)
        del result[-1]
        result.append(currentStepData[0])
        if include_delimiters:
            result.append(currentStepData[1])
        result.append(currentStepData[2])
    if include_empty:
        return result
    else:
        return [item for item in result if len(item) > 0]

assert chain_string_partitions("a0b1c2d3e4f5", "abcdf") == ['', 'a', '0', 'b', '1', 'c', '2', 'd', '3e4', 'f', '5']
assert chain_string_partitions("a0b1c2d3e4f5", "abcdf", include_delimiters=False) == ['', '0', '1', '2', '3e4', '5']
assert chain_string_partitions("a0b1c2d3e4f5", "a5", include_delimiters=True) == ['', 'a', '0b1c2d3e4f', '5', '']
assert chain_string_partitions("a0b1c2d3e4f5", "a5", include_delimiters=True, include_empty=False) == ['a', '0b1c2d3e4f', '5']









def path_exists(path):
    return os.access(path, 0, follow_symlinks=False)
    
    
def path_is_populated(path):
    return os.access(path, 0, follow_symlinks=True)


def existent_path_last_element_is_symlink(path):
    if not path_exists(path):
        raise ValueError("nothing (no link or file) exists at path={}.".format(repr(path)))
        
    try:
        testResult = os.readlink(path)
        # assert os.access(testResult, 0, follow_symlinks=False) == True, ("link is broken", path, testResult)
        return True
    except OSError as ose:
        return False
    except FileNotFoundError as fnfe:
        raise fnfe
        
    assert False
    

def existent_path_last_element_is_concrete(path):
    if not path_exists(path):
        raise ValueError("nothing exists at path={}.".format(repr(path)))
    
    isSymlink = existent_path_last_element_is_symlink(path)
    if not isSymlink:
        assert path_is_populated(path)
    return not isSymlink
    


def readlink_or_passthrough(path, recursive=False):
    try:
        result = os.readlink(path)
    except OSError as ose:
        return path
    except FileNotFoundError as fnfe:
        return path
        
    if recursive:
        return readlink_or_passthrough(result, recursive=True)
    else:
        return result


def paths_have_same_target(path0, path1):
    return (readlink_or_passthrough(path0, recursive=True) == readlink_or_passthrough(path1, recursive=True))



def safe_make_symlink(src, dst, allow_rewrite_symlink=False):
    # make symlink without overwriting anything.
    # dst is the name of the output file, which is a link. the link will point to the path given by the argument src.
    # ( DATA <- src        ) becomes ( DATA <- src <- dst ).
    if src == dst:
        raise ValueError("src and dst should not be equal!")
    # loops longer than one are NOT tested for by this method.
    
    if os.access(dst, 0, follow_symlinks=False):
        if path_last_element_is_symlink(dst):
            if allow_rewrite_symlink:
                warningMessageStart = "warning: to create this new link:\n  {} -> {}".format(repr(dst), repr(src))
                linkWillNotChange = (os.readlink(dst) == src)
                print("{}\nthis method will overwrite the existing link:\n  {} -> {}.\n{}.".format(
                        warningMessageStart, repr(dst), repr(os.readlink(dst)),
                        "This will not change anything" if linkWillNotChange else "This is a different target",
                    ))
                os.unlink(dst)
                os.symlink(src, dst)
            else:
                raise ValueError("cannot continue. The thing at the last element of dst={} is a symlink, but allow_rewrite_symlink is not True.".format(dst))
        else:
            raise ValueError("cannot continue. There is something concrete at dst={}. This method only works when dst does not exist, or when dst's last element is a symlink and allow_rewrite_symlink=True.".format(repr(dst)))
    
    else:
        os.symlink(src, dst)


def safe_substitute_with_symlink(local, remote):
    # move from local to remote, then at local, put a link to remote.
    # ( DATA <-------- local ) becomes ( DATA <- remote <- local ).
    if local == remote:
        raise ValueError("will not create a link that links directly to itself.")
    if readlink_or_passthrough(local) == remote:
        raise ValueError("the specified link already exists.")
    if readlink_or_passthrough(local) == readlink_or_passthrough(remote):
        raise ValueError("will not create a circular link or a link that already exists.")
    if os.access(local, 0, follow_symlinks=False):
        if existent_path_last_element_is_symlink(local):
            raise NotImplementedError("local is already a symlink to something. chaining symlinks is not supported yet, as it might not be safe - the depth is not checked and could grow too deep.")
    
    assert path_exists(local), "this can't be done because the thing does not exist."
    assert not path_exists(remote), "this can't be done because a file already exists at the path remote (the place where the file will be moved to, and the link will point to)."
    
    # os.replace(local, remote) # like mv. but can't work across two file systems.
    shutil.move(local, remote)
    
    assert not path_exists(local), "move failed somehow."
    assert path_exists(remote), "move failed somehow."
    
    safe_make_symlink(remote, local)
    
    assert path_exists(local)
    assert path_exists(remote)
    assert paths_have_same_target(local, remote)


def remotify(local, remote):
    # whichever is applicable will apply:
    #   ( DATA <-------- local ) becomes ( DATA <- remote <- local )
    #   ( (EMPTY remote); (EMPTY local) ) becomes ( (EMPTY remote) <- local )
    if path_exists(remote):
        raise ValueError("cannot continue. there is something at remote={}.".format(repr(remote)))
        
    if path_exists(local):
        if existent_path_last_element_is_symlink(local):
            raise NotImplementedError("can't remotify a symlink yet, depth is not regulated.")
        safe_substitute_with_symlink(local, remote)
    else:
        safe_make_symlink(remote, local)
    assert path_exists(local)
    assert path_exists(remote)
    assert paths_have_same_target(local, remote)


def match_highest_dict_key_not_higher(data_dict, key):
    if key in data_dict:
        return data_dict[key]
    if not len(data_dict) > 0:
        raise KeyError("empty data dict.")
    bestKey = None
    for dataKey in data_dict.keys():
        assert isinstance(dataKey, type(key))
        if dataKey > key:
            continue
        else:
            assert dataKey < key
            if bestKey is None or dataKey > bestKey:
                bestKey = dataKey
    assert bestKey is not None
    return data_dict[bestKey]
    
assert match_highest_dict_key_not_higher({5:"a",10:"b",3:"c",20:"d"}, 9) == "a"
assert match_highest_dict_key_not_higher({5:"a",10:"b",3:"c",20:"d"}, 10) == "b"
        

class MinecraftRegionNames:
    REGION_NAME_TEMPLATES = {(1,6,0):"r.{}.{}.mca"}
    def __init__(self, version_string):
        assert version_string.count(".") == 2
        self.version = tuple(version_string.split("."))
        self.region_name_template = match_highest_dict_key_not_higher(REGION_NAME_TEMPLATES, self.version)
    def coords_to_name(self, coords):
        return self.region_name_template.format(*coords)
    def name_to_coords(self, name):
        result = chain_string_partitions(name, self.region_name_template.split("{}"), include_delimiters=False, include_empty=False)
        assert len(result) == self.region_name_template.count("{}"), "invalid or unknown format for region name {}.".format(name)
        return tuple(int(item) for item in result)




"""
def convert_world_first_time():
    ...
"""

def ensure_ends_with(string0, string1):
    if string0.endswith(string1):
        return string0
    else:
        return string0 + string1
        
def join_without_repetition(string0, string1, sep=None):
    assert not string1.startswith(sep)
    return ensure_ends_with(string0, sep) + string1


def exampleCheckerboardNameCategorizer(region_name):
    return sum(Minecraft.region_name_to_coords(region_name))%2


def exampleCheckerboardCoordsCategorizer(coords):
    return sum(coords)%2
    
"""

    for categoryPath in category_paths:
        if not path_exists(categoryPath):
            raise ValueError("category path {} does not exist.".format(categoryPath))
            
    print("there are {} non-directory objects in {}.".format(len(regionFileNames), repr(regions_path)))
    # remotifiedCount, skippedCount = (0, 0)
    # skippedConformantCount, skippedNonconformantCount = (0, 0)
    # print("done. remotified: {}. skipped: {}.".format(remotifiedCount, skippedCount))
            
"""

def dirlist_files(path):
    # this includes broken symlinks and symlinks to files.
    return list(os.walk(path))[0][2]


def dirlist_files_where(path, key_fun):
    result = []
    for filename in dirlist_files(path):
        filepath = ensure_ends_with(path, os.sep) + filename
        if not path_exists(filepath):
            print("Warning: the file {} seems to have disappeared.")
            continue
        if key_fun(filepath):
            result.append(filename)
    return result
    
"""
def dirlist_broken_symlinks(path):
    return dirlist_files_where(path, (lambda filepath: not path_is_populated(filepath)))
    
def dirlist_symlink_files(path):
    return dirlist_files_where(path, existent_path_last_element_is_symlink)
    
def dirlist_nonsymlink_files(path):
    return dirlist_files_where(path, (lambda filepath: not existent_path_last_element_is_symlink(filepath)))
"""

def dirlist_populated(path):
    # list filepaths that direct to a file that exists.
    return dirlist_files_where(path, path_is_populated)


def convert_whole_world(regions_path, coords_categorizer, category_paths, region_name_scheme=MinecraftRegionNames):
    
    regionFileNames = dirlist_files(regions_path)
    
    for regionFileName in regionFileNames:
        regionFilePath = join_without_repetition(regions_path, regionFileName, sep=os.sep)
        regionFileCoords = region_name_scheme.name_to_coords(regionFileName)
        regionFileCategory = coords_categorizer(regionFileCoords)
        
        print("region {} is category {}.".format(regionFileName, regionFileCategory))
        
        regionFileRemotePath = join_without_repetition(category_paths[regionFileCategory], regionFileName, sep=os.sep)
        if existent_path_last_element_is_concrete(regionFilePath):
            print("  it will be remotified to {}.".format(regionFileRemotePath))
            safe_substitute_with_symlink(regionFilePath, regionFileRemotePath)
        else:
            print("  it is not concrete (it is probably a symlink) and it will be skipped.")
            if not paths_have_same_target(regionFilePath, regionFileRemotePath):
                print("  it DOES NOT conform to the categorizer.")
    

def future_proof_world(regions_path, coords_categorizer, category_paths, region_name_scheme=MinecraftRegionNames, region_steps=1):

    populatedNames = dirlist_populated(regions_path)
    
    populatedCoordPairSet = set(region_name_scheme.name_to_coords(name) for name in populatedNames)
    stageSets = get_set_expansion_stages_using_multi_mutation(populatedCoordPairSet, gen_directly_adjacent_int_vecs, region_steps)
    print("prepared {} stages, totaling {} items.".format(len(stageSets), sum(len(item) for item in stageSets)))
    for stageSet in stageSets:
        for newRegionCoords in stageSet:
            newRegionCategory = coords_categorizer(newRegionCoords)
            newRegionName = region_name_scheme.coords_to_name(newRegionCoords)
            newRegionLocalPath = join_without_repetition(regions_path, newRegionName, sep=os.sep)
            newRegionRemotePath = join_without_repetition(category_paths[newRegionCategory], newRegionName, sep=os.sep)
            print(newRegionName + ":")
            if path_exists(newRegionLocalPath):
                print("  already exists,")
                if path_is_populated(newRegionLocalPath):
                    print("  has been populated while this program is working.")
                print("  it will be skipped.")
                continue
            else:
                print("  will be processed.")
                safe_make_symlink(newRegionRemotePath, newRegionLocalPath)
    print("done")
                
        

    
def revert_world():
    raise NotImplementedError()


    
    
    
    
class Bash:
    file_exists_and_is = {
            "block_special":"b", "character_special":"c", "directory":"d", "generic":"e",
            "regular_file":"f", "symbolic_link":"h", "named_pipe":"p", "not_empty":"s",
            "socket":"S",
        }
    file_exists_and_permits = {"read":"r", "write":"w", "execute":"x"}