package lukai

import (
	"context"
	"fmt"
	"math/rand"

	"github.com/luk-ai/lukai/protobuf/aggregatorpb"
	"github.com/pkg/errors"
)

// logError just logs an error without reporting it.
func (mt *ModelType) logError(id aggregatorpb.ModelID, typ aggregatorpb.ErrorType, err error) {
	mt.errors.Lock()
	defer mt.errors.Unlock()

	pberr := aggregatorpb.Error{
		Id:    id,
		Type:  typ,
		Error: fmt.Sprintf("%+v", err),
	}

	if len(mt.errors.errors) < MaxQueuedErrors {
		mt.errors.errors = append(mt.errors.errors, pberr)
	} else {
		// If we've hit the max queued errors, randomly replace one of the earlier
		// errors.
		mt.errors.errors[rand.Intn(len(mt.errors.errors))] = pberr
	}
}

// reportError reports an error to the edge client.
func (mt *ModelType) reportError(ctx context.Context, edge aggregatorpb.EdgeClient, id aggregatorpb.ModelID, typ aggregatorpb.ErrorType, err error) error {
	mt.logError(id, typ, err)

	if !mt.errorLimiter.Allow() {
		return nil
	}

	return mt.reportErrorsInternal(ctx, edge)
}

// reportErrorDial reports an error by dialing a new edge client.
func (mt *ModelType) reportErrorDial(ctx context.Context, id aggregatorpb.ModelID, typ aggregatorpb.ErrorType, err error) error {
	mt.logError(id, typ, err)

	if !mt.errorLimiter.Allow() {
		return nil
	}

	conn, err := dial(ctx, EdgeAddress)
	if err != nil {
		return err
	}
	defer conn.Close()

	edge := aggregatorpb.NewEdgeClient(conn)

	return mt.reportErrorsInternal(ctx, edge)
}

// reportErrorsInternal actually sends the errors to the server. Should use
// reportError or reportErrorDial instead.
func (mt *ModelType) reportErrorsInternal(ctx context.Context, edge aggregatorpb.EdgeClient) error {
	mt.errors.Lock()
	defer mt.errors.Unlock()

	if _, err := edge.ReportError(ctx, &aggregatorpb.ReportErrorRequest{
		Errors: mt.errors.errors,
	}); err != nil {
		return errors.Wrapf(err, "failed to report error")
	}

	mt.errors.errors = mt.errors.errors[:0]

	return nil
}
