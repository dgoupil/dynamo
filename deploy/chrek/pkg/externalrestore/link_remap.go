package externalrestore

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/checkpoint-restore/go-criu/v8/crit"
	"github.com/checkpoint-restore/go-criu/v8/crit/images/fdinfo"
	"github.com/checkpoint-restore/go-criu/v8/crit/images/regfile"
	remap_file_path "github.com/checkpoint-restore/go-criu/v8/crit/images/remap-file-path"
	"github.com/sirupsen/logrus"
	"google.golang.org/protobuf/proto"
)

type fileInfo struct {
	name string
	mode os.FileMode
}

type remapEntry struct {
	origID  uint32
	remapID uint32
}

func createLinkRemapStubs(checkpointPath, targetRoot string, log *logrus.Entry) error {
	remapPath := filepath.Join(checkpointPath, "remap-fpath.img")
	remaps, err := parseRemapFpath(remapPath)
	if err != nil {
		if os.IsNotExist(err) {
			log.Debug("No remap-fpath.img found, no link_remap stubs needed")
			return nil
		}
		return fmt.Errorf("failed to parse remap-fpath.img: %w", err)
	}
	if len(remaps) == 0 {
		return nil
	}

	regFilesPath := filepath.Join(checkpointPath, "reg-files.img")
	filesPath := filepath.Join(checkpointPath, "files.img")

	fileMap, parseErr := parseRegFilesWithMode(regFilesPath)
	if parseErr != nil {
		fileMap, parseErr = parseFilesImgWithMode(filesPath)
		if parseErr != nil {
			return fmt.Errorf("failed to parse files metadata for remap stubs: %w", parseErr)
		}
	}

	created := 0
	for _, remap := range remaps {
		origInfo, ok := fileMap[remap.origID]
		if !ok {
			continue
		}

		remapPathInContainer := ""
		remapMode := origInfo.mode
		if remapInfo, ok := fileMap[remap.remapID]; ok {
			remapPathInContainer = remapInfo.name
			remapMode = remapInfo.mode
		} else {
			dir := filepath.Dir(origInfo.name)
			if !strings.HasPrefix(dir, "/") {
				dir = "/" + dir
			}
			remapPathInContainer = filepath.Join(dir, fmt.Sprintf("link_remap.%d", remap.remapID))
		}

		if !strings.HasPrefix(remapPathInContainer, "/") {
			remapPathInContainer = "/" + remapPathInContainer
		}

		hostPath := filepath.Join(targetRoot, strings.TrimPrefix(filepath.Clean(remapPathInContainer), "/"))
		if _, err := os.Stat(hostPath); err == nil {
			continue
		}

		if err := createLinkRemapStub(hostPath, remapMode); err != nil {
			log.WithError(err).WithFields(logrus.Fields{
				"stub":   hostPath,
				"target": origInfo.name,
			}).Warn("Failed to create link_remap stub")
			continue
		}
		created++
	}

	if created > 0 {
		log.WithField("count", created).Info("Created link_remap stubs")
	}
	return nil
}

func parseRemapFpath(path string) ([]remapEntry, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	magic, err := crit.ReadMagic(f)
	if err != nil {
		return nil, fmt.Errorf("failed to read remap image magic: %w", err)
	}
	if magic != "REMAP_FPATH" {
		return nil, fmt.Errorf("unexpected remap image magic: %s", magic)
	}

	sizeBuf := make([]byte, 4)
	var entries []remapEntry
	for {
		_, err := io.ReadFull(f, sizeBuf)
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("failed to read remap entry size: %w", err)
		}

		entrySize := binary.LittleEndian.Uint32(sizeBuf)
		entryBuf := make([]byte, entrySize)
		if _, err := io.ReadFull(f, entryBuf); err != nil {
			return nil, fmt.Errorf("failed to read remap entry: %w", err)
		}

		entry := &remap_file_path.RemapFilePathEntry{}
		if err := proto.Unmarshal(entryBuf, entry); err != nil {
			return nil, fmt.Errorf("failed to decode remap entry: %w", err)
		}
		entries = append(entries, remapEntry{
			origID:  entry.GetOrigId(),
			remapID: entry.GetRemapId(),
		})
	}
	return entries, nil
}

func parseRegFilesWithMode(path string) (map[uint32]fileInfo, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	magic, err := crit.ReadMagic(f)
	if err != nil {
		return nil, fmt.Errorf("failed to read reg-files image magic: %w", err)
	}
	if magic != "REG_FILES" {
		return nil, fmt.Errorf("unexpected reg-files image magic: %s", magic)
	}

	sizeBuf := make([]byte, 4)
	fileMap := make(map[uint32]fileInfo)
	for {
		_, err := io.ReadFull(f, sizeBuf)
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("failed to read reg-files entry size: %w", err)
		}

		entrySize := binary.LittleEndian.Uint32(sizeBuf)
		entryBuf := make([]byte, entrySize)
		if _, err := io.ReadFull(f, entryBuf); err != nil {
			return nil, fmt.Errorf("failed to read reg-files entry: %w", err)
		}

		entry := &regfile.RegFileEntry{}
		if err := proto.Unmarshal(entryBuf, entry); err != nil {
			return nil, fmt.Errorf("failed to decode reg-files entry: %w", err)
		}

		mode := os.FileMode(entry.GetMode() & 0777)
		if mode == 0 {
			mode = 0600
		}

		fileMap[entry.GetId()] = fileInfo{
			name: entry.GetName(),
			mode: mode,
		}
	}
	return fileMap, nil
}

func parseFilesImgWithMode(path string) (map[uint32]fileInfo, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	magic, err := crit.ReadMagic(f)
	if err != nil {
		return nil, fmt.Errorf("failed to read files image magic: %w", err)
	}
	if magic != "FILES" {
		return nil, fmt.Errorf("unexpected files image magic: %s", magic)
	}

	sizeBuf := make([]byte, 4)
	fileMap := make(map[uint32]fileInfo)
	for {
		_, err := io.ReadFull(f, sizeBuf)
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("failed to read files entry size: %w", err)
		}

		entrySize := binary.LittleEndian.Uint32(sizeBuf)
		entryBuf := make([]byte, entrySize)
		if _, err := io.ReadFull(f, entryBuf); err != nil {
			return nil, fmt.Errorf("failed to read files entry: %w", err)
		}

		entry := &fdinfo.FileEntry{}
		if err := proto.Unmarshal(entryBuf, entry); err != nil {
			return nil, fmt.Errorf("failed to decode files entry: %w", err)
		}
		if entry.GetReg() == nil {
			continue
		}

		mode := os.FileMode(entry.GetReg().GetMode() & 0777)
		if mode == 0 {
			mode = 0600
		}

		fileMap[entry.GetId()] = fileInfo{
			name: entry.GetReg().GetName(),
			mode: mode,
		}
	}
	return fileMap, nil
}

func createLinkRemapStub(path string, mode os.FileMode) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write(make([]byte, 32))
	return err
}
