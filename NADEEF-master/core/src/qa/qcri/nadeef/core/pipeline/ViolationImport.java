/*
 * QCRI, NADEEF LICENSE
 * NADEEF is an extensible, generalized and easy-to-deploy data cleaning platform built at QCRI.
 * NADEEF means "Clean" in Arabic
 *
 * Copyright (c) 2011-2013, Qatar Foundation for Education, Science and Community Development (on
 * behalf of Qatar Computing Research Institute) having its principle place of business in Doha,
 * Qatar with the registered address P.O box 5825 Doha, Qatar (hereinafter referred to as "QCRI")
 *
 * NADEEF has patent pending nevertheless the following is granted.
 * NADEEF is released under the terms of the MIT License, (http://opensource.org/licenses/MIT).
 */

package qa.qcri.nadeef.core.pipeline;

import com.google.common.base.Optional;
import qa.qcri.nadeef.core.datamodel.NadeefConfiguration;
import qa.qcri.nadeef.core.datamodel.Rule;
import qa.qcri.nadeef.core.datamodel.Violation;
import qa.qcri.nadeef.core.utils.Violations;
import qa.qcri.nadeef.core.utils.sql.DBConnectionPool;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.Collection;

/**
 * Import violations from violation table.
 */
public class ViolationImport extends Operator<Optional, Collection<Violation>> {
    ViolationImport(ExecutionContext context) {
        super(context);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Collection<Violation> execute(Optional empty) throws Exception {
        ResultSet resultSet = null;
        Collection<Violation> result = null;
        DBConnectionPool connectionPool = getCurrentContext().getConnectionPool();
        Rule rule = getCurrentContext().getRule();
        try (
            Connection conn = connectionPool.getNadeefConnection();
            Statement stat = conn.createStatement();
        ) {
            conn.setAutoCommit(true);
            resultSet = stat.executeQuery(
                "SELECT * FROM " +
                    NadeefConfiguration.getViolationTableName() +
                    " WHERE RID = '" +
                    rule.getRuleName() +
                    "' ORDER BY vid"
            );

            result = Violations.fromQuery(resultSet);
        } finally {
            if (resultSet != null) {
                resultSet.close();
            }
        }
        return result;
    }
}
